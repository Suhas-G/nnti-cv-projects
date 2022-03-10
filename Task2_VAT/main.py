#!/usr/bin/env python3
import argparse
import datetime
import math
import random

import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from torch.utils.tensorboard import SummaryWriter

torch.manual_seed(42)
random.seed(42)
np.random.seed(42)

from dataloader import get_cifar10, get_cifar100
from model.wrn import WideResNet
from utils import accuracy, load_checkpoint, save_checkpoint
from vat import VATLoss

current_time = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M")

def create_validation_dataset(unlabelled_train: torch.utils.data.Dataset, num_classes: int, size=1000):
    ''' Create a validation dataset from the unlabelled dataset.
        It creates a balanced dataset of size `size` by choosing equal images from each classes.
    '''
    class_size = size // num_classes
    validation_idx = []
    for c in range(num_classes):
        c_targets = np.where(unlabelled_train.targets == c)[0]
        # Sanity check to make sure we have enough data from each class
        assert len(c_targets) >= class_size
        idx = np.random.choice(c_targets, class_size, replace=False)
        validation_idx.extend(idx)
    
    return Subset(unlabelled_train, validation_idx)

def test(model: torch.nn.Module, loader: DataLoader, 
         criterion: torch.nn.CrossEntropyLoss, device: torch.device):
    ''' Calculate loss and accuracy for given data
    '''
    model.eval()
    val_loss = 0.0
    total = 0.0
    correct = 0.0
    with torch.no_grad():
        for data, target in loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            val_loss += criterion(output, target).item()
            _, predicted = torch.max(output.data, 1)
            total += target.size(0)
            correct += (predicted == target).sum().item()
    return val_loss / len(loader), (correct / total) * 100

def get_params(args: argparse.Namespace):
    '''Helper function to get all the parameters used for the run'''
    return {
        'dataset': args.dataset,
        'num_labeled': args.num_labeled,
        'lr': args.lr,
        'weight_decay': args.wd,
        'train_labelled_batch_size': args.train_labelled_batch,
        'train_unlabelled_batch_size': args.train_unlabelled_batch,
        'test_batch_size': args.test_batch,
        'total_iter': args.total_iter,
        'iter_per_epoch': args.iter_per_epoch,
        'model_depth': args.model_depth,
        'model_width': args.model_width,
        'vat_xi': args.vat_xi,
        'vat_eps': args.vat_eps,
        'vat_alpha': args.vat_alpha,
        'vat_iter': args.vat_iter
    }

def get_next_batch(dataset: torch.utils.data.Dataset, loader: DataLoader, 
                   batch_size: int, num_workers: int, shuffle=True):
    '''
    Helper function to get the next batch of data from the loader.
    If the loader is exhausted, it will be reinitialised.
    '''
    try:
        x, y = next(loader)
    except StopIteration:
        loader = iter(DataLoader(dataset, batch_size=batch_size, 
                                shuffle=shuffle, num_workers=num_workers))
        x, y = next(loader)

    return x, y, loader


def main(args):
    '''Main train and test function'''
    if args.dataset == "cifar10":
        args.num_classes = 10
        labeled_dataset, unlabeled_dataset, test_dataset = get_cifar10(args, 
                                                                args.datapath)
    if args.dataset == "cifar100":
        args.num_classes = 100
        labeled_dataset, unlabeled_dataset, test_dataset = get_cifar100(args, 
                                                                args.datapath)

    validation_dataset = create_validation_dataset(unlabeled_dataset, args.num_classes, size=10000)
    args.epoch = math.ceil(args.total_iter / args.iter_per_epoch)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('Device: ', device)

    writer = SummaryWriter(log_dir=args.dataout)

    labeled_loader      = iter(DataLoader(labeled_dataset, 
                                    batch_size = args.train_labelled_batch, 
                                    shuffle = True, 
                                    num_workers=args.num_workers))
    unlabeled_loader    = iter(DataLoader(unlabeled_dataset, 
                                    batch_size=args.train_unlabelled_batch,
                                    shuffle = True, 
                                    num_workers=args.num_workers))
    test_loader         = DataLoader(test_dataset,
                                    batch_size = args.test_batch,
                                    shuffle = False, 
                                    num_workers=args.num_workers)
    val_loader          = DataLoader(validation_dataset, batch_size=args.test_batch,
                                    shuffle = True, num_workers=args.num_workers)
    
    model       = WideResNet(args.model_depth, 
                                args.num_classes, widen_factor=args.model_width)

    start_epoch = 0
    if args.resume:
        rets = load_checkpoint(args.model_path, model = model, return_optimiser=True)
        start_epoch = rets['epoch'] + 1
    model       = model.to(device)

    ############################################################################
    # TODO: SUPPLY your code
    ############################################################################

    optimiser = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.wd)
    if args.resume:
        optimiser.load_state_dict(rets['optimiser'])
    criterion = torch.nn.CrossEntropyLoss()
    vat_criterion = VATLoss(args)
    
    for epoch in range(start_epoch, start_epoch + args.epoch):
        model.train()
        for i in range(args.iter_per_epoch):
            x_l, y_l, labeled_loader = get_next_batch(labeled_dataset, labeled_loader, 
                                                        args.train_labelled_batch, args.num_workers)
            
            x_ul, _, unlabeled_loader = get_next_batch(unlabeled_dataset, unlabeled_loader, 
                                                        args.train_unlabelled_batch, args.num_workers)

            # Combined labelled and unlabelled data. Choose batch size samples randomly.
            _combined_size = x_l.size()[0] + x_ul.size()[0]
            # If combined size is lesser than the batch size, take all of it
            sampling_indices = np.random.choice(_combined_size, min(args.train_unlabelled_batch, _combined_size), replace = False)
            x_star = torch.cat((x_l, x_ul), dim=0)[sampling_indices]
            
            x_l, y_l    = x_l.to(device), y_l.to(device)
            x_star      = x_star.to(device)
            ####################################################################
            # TODO: SUPPLY you code
            ####################################################################

            optimiser.zero_grad()

            vat_loss = vat_criterion(model, x_star)

            outputs_labelled = model(x_l)
            loss_labelled = criterion(outputs_labelled, y_l)
            # Total loss is a weighted sum of labelled loss and VAT loss
            total_loss = loss_labelled + args.vat_alpha * vat_loss
            total_loss.backward()
            optimiser.step()


            writer.add_scalar('loss/train_labelled', loss_labelled, epoch*args.iter_per_epoch+i)
            writer.add_scalar('loss/vat', vat_loss, epoch*args.iter_per_epoch+i)
            writer.add_scalar('loss/total', total_loss.item(), epoch*args.iter_per_epoch+i)

        # Validation once per epoch
        validation_loss, validation_accuracy = test(model, val_loader, criterion, device)
        writer.add_scalar('loss/validation', validation_loss, epoch)
        writer.add_scalar('accuracy/validation', validation_accuracy, epoch)

        # Save checkpoint for initial and last epoch as well at regular intervals
        if epoch == 0 or (epoch + 1) % args.checkpoint_freq == 0 or epoch == args.epoch - 1:
            save_checkpoint(model, epoch, './checkpoints/{}/checkpoint_{}.pth'.format(current_time,epoch + 1),
                            optimiser = optimiser, params=get_params(args))


    test_loss, test_accuracy = test(model, test_loader, criterion, device)
    # Add all hyper parameters to the tensorboard, along with validation metrics, which is used to compare.
    writer.add_hparams( get_params(args),
                        {
                            'final_validation_loss': validation_loss,
                            'final_validation_accuracy': validation_accuracy,
                        })
    writer.add_scalar('loss/test', test_loss, epoch)
    writer.add_scalar('accuracy/test', test_accuracy, epoch)
    writer.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Virtual adversarial training \
                                        of CIFAR10/100 using with pytorch")

    # Dataset parameters
    parser.add_argument("--dataset", default="cifar10", 
                        type=str, choices=["cifar10", "cifar100"])
    parser.add_argument("--datapath", default="./data/", 
                        type=str, help="Path to the CIFAR-10/100 dataset")
    parser.add_argument('--num-labeled', type=int, 
                        default=4000, help='Total number of labeled samples')
    parser.add_argument("--expand-labels", action="store_true", 
                        help="expand labels to fit eval steps")
    parser.add_argument('--train-labelled-batch', default=64, type=int,
                        help='Batchsize for labelled train data')
    parser.add_argument('--train-unlabelled-batch', default=256, type=int,
                        help='Batchsize for unlabelled train data')
    parser.add_argument('--test-batch', default=64, type=int,
                        help='Batch size for test data')
    parser.add_argument('--num-workers', default=1, type=int,
                        help="Number of workers to launch during training")

    # Model parameters
    parser.add_argument("--model-depth", type=int, default=28,
                        help="model depth for wide resnet") 
    parser.add_argument("--model-width", type=int, default=2,
                        help="model width for wide resnet")

    # Optimisation parameters
    parser.add_argument("--lr", default=0.001, type=float, 
                        help="The initial learning rate") 
    parser.add_argument("--wd", default=0, type=float,
                        help="Weight decay")


    # Training Control parameters
    parser.add_argument('--total-iter', default=128*375, type=int,
                        help='total number of iterations to run')
    parser.add_argument('--iter-per-epoch', default=375, type=int,
                        help="Number of iterations to run per epoch")
    parser.add_argument("--checkpoint-freq", type=int, default=20, 
                        help="Save checkpoint after these many epochs")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--model-path", type=str)


    # Logging parameters
    parser.add_argument("--dataout", type=str, default="./runs/{}".format(current_time),
                        help="Path to save log files")
    

    # VAT parameters
    parser.add_argument("--vat-xi", default=1, type=float, 
                        help="VAT xi parameter")
    parser.add_argument("--vat-eps", default=2.0, type=float, 
                        help="VAT epsilon parameter") 
    parser.add_argument('--vat-alpha', type=float, default=1.0, metavar='ALPHA',
                        help='VAT regularization coefficient (default: 0.01)')
    parser.add_argument("--vat-iter", default=1, type=int, 
                        help="VAT iteration parameter") 
    
    args = parser.parse_args()

    args.train_batch = args.train_labelled_batch
    if args.resume and args.model_path is None:
        parser.error("--resume requires --model-path."
                    "If resuming from previous checkpoint, provide checkpoint path")

    main(args)
