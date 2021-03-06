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
from utils import load_checkpoint, save_checkpoint

current_time = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M")

class PseudoLabelledDataset(torch.utils.data.Dataset):
    '''Dataset to hold pseudo labelled data
       It keeps the last `max_length` elements appended.
       The `data` is a tensor of shape (N, C, H, W) after transformations.
       The `targets` is a tensor of shape (N,)
    '''
    def __init__(self) -> None:
        super().__init__()
        self.data = None
        self.targets = None
        self.max_length = 100000

    def __len__(self) -> int:
        ''' Returns the number of elements in the dataset.
            0 if not yet initialised.
        '''
        if self.data is None:
            return 0
        return len(self.data)

    def __getitem__(self, idx: int):
        '''Returns the data (image) and target class at index `idx`
        '''
        return self.data[idx], self.targets[idx]

    def append(self, data: torch.tensor, target: torch.tensor):
        ''' Append the data and target to the dataset.
            If the dataset is not initialised, initialise it.
            If the dataset size is more than `max_length`, truncate the 
            extra elements from the beginning of array
        '''
        if self.data is None:
            self.data = data.detach().clone()
            self.targets = target.detach().clone()
        else:
            self.data = torch.cat([self.data, data.detach().clone()])
            self.targets = torch.cat([self.targets, target.detach().clone()])

        # Truncate if exceeding the limit. This is one way to keep the run within memory limits.
        if len(self.data) > self.max_length:
            self.data = self.data[-self.max_length:]
            self.targets = self.targets[-self.max_length:]

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
        'lr_decay': args.lr_decay,
        'momentum': args.momentum,
        'train_batch_size': args.train_batch,
        'test_batch_size': args.test_batch,
        'total_iter': args.total_iter,
        'iter_per_epoch': args.iter_per_epoch,
        'threshold': args.threshold,
        'model_depth': args.model_depth,
        'model_width': args.model_width,
        'pre_train': args.pre_train
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

def get_pseudo_loss_coeff(epoch: int, max_coeff: float, pre_trained: int, total_epochs: int):
    ''' Calculate the pseudo loss coefficient for the given epoch.
        Initially when pre-training (or warmup) pseudo loss coefficient is 0 as no data is
        added to `PseudoLabelledDataset`. After that, it increases linearly from 0 to `max_coeff`
        until half of total epochs. And then stays constant. 
        This implementation loosely follows the paper:
        'Pseudo-Label: The Simple and Efficient Semi-Supervised Learning Method for Deep Neural Networks'
    '''
    if epoch < pre_trained:
        return 0.0
    elif epoch < total_epochs // 2:
        return ((epoch - pre_trained) / ((total_epochs // 2) - pre_trained)) * max_coeff
    else:
        return max_coeff

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
    pseudo_labelled = PseudoLabelledDataset()
    validation_dataset = create_validation_dataset(unlabeled_dataset, args.num_classes, size=10000)
    args.epoch = math.ceil(args.total_iter / args.iter_per_epoch)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('Device: ', device)

    writer = SummaryWriter(log_dir=args.dataout)

    labeled_loader      = iter(DataLoader(labeled_dataset, 
                                    batch_size = args.train_batch, 
                                    shuffle = True, 
                                    num_workers=args.num_workers))
    unlabeled_loader    = iter(DataLoader(unlabeled_dataset, 
                                    batch_size=args.train_batch,
                                    shuffle = True, 
                                    num_workers=args.num_workers))
    test_loader         = DataLoader(test_dataset,
                                    batch_size = args.test_batch,
                                    shuffle = False, 
                                    num_workers=args.num_workers)
    val_loader          = DataLoader(validation_dataset, batch_size=args.test_batch,
                                    shuffle = True, num_workers=args.num_workers)
    pseudo_loader       = None
    
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

    optimiser = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum)
    if args.resume:
        optimiser.load_state_dict(rets['optimiser'])
    criterion = torch.nn.CrossEntropyLoss()
    scheduler = optim.lr_scheduler.ExponentialLR(optimiser, gamma=args.lr_decay)
    
    for epoch in range(start_epoch, start_epoch + args.epoch):
        model.train()
        
        for i in range(args.iter_per_epoch):

            x_l, y_l, labeled_loader = get_next_batch(labeled_dataset, labeled_loader, 
                                                        args.train_batch, args.num_workers)
            x_ul, _, unlabeled_loader = get_next_batch(unlabeled_dataset, unlabeled_loader, 
                                                        args.train_batch, args.num_workers)
            x_l, y_l = x_l.to(device), y_l.to(device)
            x_ul = x_ul.to(device)

            if pseudo_loader is not None:
                x_pseudo, y_pseudo, pseudo_loader = get_next_batch(pseudo_labelled, pseudo_loader, 
                                                                    args.train_batch, args.num_workers)
                x_pseudo, y_pseudo = x_pseudo.to(device), y_pseudo.to(device)

            ####################################################################
            # TODO: SUPPLY your code
            ####################################################################
            optimiser.zero_grad()
            pseudo_loss = 0.0
            outputs_labelled = model(x_l)
            labelled_size = outputs_labelled.size(0)
            # This is just a small number to prevent division by 0
            pseudo_size = 1e-8

            # If pseudo labelled dataset is initialised, it means pre-train stage is done
            # So get the loss for the outputs
            if pseudo_loader is not None:
                outputs_pseudo_labelled = model(x_pseudo)
                pseudo_size = outputs_pseudo_labelled.size(0)
                pseudo_loss = get_pseudo_loss_coeff(epoch, args.pseudo_loss_coeff, args.pre_train, start_epoch + args.epoch) * criterion(outputs_pseudo_labelled, y_pseudo)

            loss_labelled = criterion(outputs_labelled, y_l)
            # Total loss is a weighted sum of labelled loss and pseudo loss
            total_loss = loss_labelled + pseudo_loss
            total_loss.backward()
            optimiser.step()

            # Normalised loss
            writer.add_scalar('loss/train_labelled', loss_labelled / labelled_size, epoch*args.iter_per_epoch+i)
            writer.add_scalar('loss/pseudo_labelled', pseudo_loss / pseudo_size, epoch*args.iter_per_epoch+i)
            writer.add_scalar('loss/total', total_loss.item() / (labelled_size + pseudo_size), epoch*args.iter_per_epoch+i)

            # If above pre-training mode, the get high confidence labels as pseudo label dataset
            if epoch >= args.pre_train:
                with torch.no_grad():
                    outputs = model(x_ul)
                    outputs = torch.softmax(outputs.detach(), dim=-1)
                    x_temp = x_ul[(outputs >= args.threshold).any(axis=1)]
                    _, label_temp = torch.max(outputs[(outputs >= args.threshold).any(axis=1)], axis=1)
                    pseudo_labelled.append(x_temp.cpu(), label_temp.cpu())


        # Only if current epoch is greater than or equal to pre-train epoch,
        # we initialse pseudo labelled dataset, as otherwise it would be empty 
        if epoch >= args.pre_train and pseudo_loader is None:
            print('Starting to use pseudo labelled data...')
            pseudo_loader = iter(DataLoader(pseudo_labelled, batch_size=args.train_batch, 
                                                shuffle=True, num_workers=args.num_workers))

        # Validation once per epoch
        validation_loss, validation_accuracy = test(model, val_loader, criterion, device)
        writer.add_scalar('loss/validation', validation_loss, epoch)
        writer.add_scalar('accuracy/validation', validation_accuracy, epoch)

        writer.add_scalar('metrics/no_of_pseudo_labbeled', len(pseudo_labelled), epoch)

        scheduler.step()
        # Save checkpoint for initial and last epoch as well at regular intervals
        if epoch == 0 or (epoch + 1) % args.checkpoint_freq == 0 or epoch == args.epoch - 1:
            save_checkpoint(model, epoch, './checkpoints/{}/checkpoint_{}.pth'.format(current_time,epoch + 1),
                            optimiser = optimiser, params=get_params(args))

    # Add all hyper parameters to the tensorboard, along with validation metrics, which is used to compare.
    writer.add_hparams( get_params(args),
                        {
                            'final_validation_loss': validation_loss,
                            'final_validation_accuracy': validation_accuracy,
                        })

    test_loss, test_accuracy = test(model, test_loader, criterion, device)
    writer.add_scalar('loss/test', test_loss, epoch)
    writer.add_scalar('accuracy/test', test_accuracy, epoch)
    writer.close()



if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description="Pseudo labeling \
                                        of CIFAR10/100 with pytorch")

    # Dataset parameters
    parser.add_argument("--dataset", default="cifar10", 
                        type=str, choices=["cifar10", "cifar100"])
    parser.add_argument("--datapath", default="./data/", 
                        type=str, help="Path to the CIFAR-10/100 dataset")
    parser.add_argument('--num-labeled', type=int, 
                        default=4000, help='Total number of labeled samples')
    parser.add_argument("--expand-labels", action="store_true", 
                        help="expand labels to fit eval steps")
    parser.add_argument('--train-batch', default=64, type=int,
                        help='train batchsize')
    parser.add_argument('--test-batch', default=64, type=int,
                        help='train batchsize')
    parser.add_argument('--num-workers', default=1, type=int,
                        help="Number of workers to launch during training")

    # Model parameters
    parser.add_argument("--model-depth", type=int, default=28,
                        help="model depth for wide resnet") 
    parser.add_argument("--model-width", type=int, default=2,
                        help="model width for wide resnet")

    # Optimisation parameters
    parser.add_argument("--lr", default=0.03, type=float, 
                        help="The initial learning rate") 
    parser.add_argument("--momentum", default=0.9, type=float,
                        help="Optimizer momentum")
    parser.add_argument("--lr-decay", default=0.95, type=float, 
                        help="Learning rate decay for Exponential scheduler")

    # Pseudo-labeling parameters
    parser.add_argument('--threshold', type=float, default=0.95,
                        help='Confidence Threshold for pseudo labeling')
    parser.add_argument('--pseudo-loss-coeff', type=float, default=2.0,
                        help='Coefficient for maximum weight for pseudo label loss')
    parser.add_argument('--pre-train', type=int, default=3, 
                        help='Number of epochs to pre-train the model before doing pseudo-labeling')


    # Training control parameters
    parser.add_argument('--total-iter', default=1024*512, type=int,
                        help='total number of iterations to run')
    parser.add_argument('--iter-per-epoch', default=1024, type=int,
                        help="Number of iterations to run per epoch")
    parser.add_argument("--checkpoint-freq", type=int, default=10, 
                        help="Save checkpoint after these many epochs")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--model-path", type=str)

    # Logging parameters
    parser.add_argument("--dataout", type=str, default="./runs/{}".format(current_time),
                        help="Path to save log files")

    
    args = parser.parse_args()

    if args.resume and args.model_path is None:
        parser.error("--resume requires --model-path."
                    "If resuming from previous checkpoint, provide checkpoint path")

    main(args)
