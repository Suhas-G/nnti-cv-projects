#!/usr/bin/env python3
import argparse
import datetime
import math
import random

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader 
from torch.utils.tensorboard import SummaryWriter


torch.manual_seed(42)
random.seed(42)
np.random.seed(42)

from dataloader import get_cifar10, get_cifar100, cifar10_mean, cifar10_std, cifar100_mean, cifar100_std
from model.wrn import WideResNet
from utils import accuracy, load_checkpoint, save_checkpoint
from loss import ContrastiveLoss, get_consistency_loss

current_time = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M")

PLOT_AUGMENTATION = False

def unnormalise(images, mean, std):
    result = images.detach().clone()
    result[:, 0] = (result[:, 0] * std[0]) + mean[0]
    result[:, 1] = (result[:, 1] * std[1]) + mean[1]
    result[:, 2] = (result[:, 2] * std[2]) + mean[2]
    return result


def get_datasets(args: argparse.Namespace):
    '''Helper function to create appropriate train, validation and test datasets'''
    if args.dataset == "cifar10":
        args.num_classes = 10
        args.mean = cifar10_mean
        args.std = cifar10_std
        labeled_dataset, unlabeled_dataset, validation_dataset, test_dataset = get_cifar10(args, 
                                                                args.datapath)
    if args.dataset == "cifar100":
        args.num_classes = 100
        args.mean = cifar100_mean
        args.std = cifar100_std
        labeled_dataset, unlabeled_dataset, validation_dataset, test_dataset = get_cifar100(args, 
                                                                args.datapath)

    return labeled_dataset, unlabeled_dataset, validation_dataset, test_dataset

def get_next_batch(dataset: torch.utils.data.Dataset, batch_size: int, num_workers: int, 
                  loader: DataLoader = None, device = torch.device('cpu'), shuffle=True):
    '''
    Helper function to get the next batch of data from the loader.
    If the loader is exhausted, it will be reinitialised.
    '''
    try:
        img, weakly_augmented, strongly_augmented, target = next(loader)
    except (StopIteration, TypeError):
        loader = iter(DataLoader(dataset, batch_size=batch_size, 
                                shuffle=shuffle, num_workers=num_workers))
        img, weakly_augmented, strongly_augmented, target = next(loader)

    img, weakly_augmented, strongly_augmented, target = (img.to(device), weakly_augmented.to(device), 
                                                         strongly_augmented.to(device), target.to(device))
    return img, weakly_augmented, strongly_augmented, target, loader

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
            output, _ = model(data)
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
        'threshold': args.threshold,
        'fixmatch_alpha': args.fixmatch_alpha,
        'contrastive_alpha': args.contrastive_alpha,
        'train_batch_size': args.train_batch,
        'test_batch_size': args.test_batch,
        'total_iter': args.total_iter,
        'iter_per_epoch': args.iter_per_epoch,
        'threshold': args.threshold,
        'model_depth': args.model_depth,
        'model_width': args.model_width
    }


def main(args: argparse.Namespace):
    labeled_dataset, unlabeled_dataset, validation_dataset, test_dataset = get_datasets(args)
    args.epoch = math.ceil(args.total_iter / args.iter_per_epoch)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('Device: ', device)

    writer = SummaryWriter(log_dir=args.dataout)

    labeled_loader = None
    unlabeled_loader = None
    test_loader = DataLoader(test_dataset, batch_size = args.test_batch,
                             shuffle = False, num_workers=args.num_workers)
    val_loader = DataLoader(validation_dataset, batch_size=args.test_batch,
                             shuffle = True, num_workers=args.num_workers)

    model = WideResNet(args.model_depth, args.num_classes, widen_factor=args.model_width)
    start_epoch = 0
    if args.resume:
        rets = load_checkpoint(args.model_path, model = model, return_optimiser=True)
        start_epoch = rets['epoch'] + 1
    model       = model.to(device)
    ce_loss_obj = nn.CrossEntropyLoss()
    contrastive_loss_obj = ContrastiveLoss(args.train_batch).to(device)

    optimiser = optim.SGD(model.parameters(), lr=args.lr, momentum=0.9)
    if args.resume:
        optimiser.load_state_dict(rets['optimiser'])

    scheduler = optim.lr_scheduler.StepLR(optimiser, step_size = 5, gamma = 0.9)
    for epoch in range(start_epoch, start_epoch + args.epoch):
        model.train()
        for i in range(args.iter_per_epoch):
            x_l, _, _, y_l, labeled_loader = get_next_batch(labeled_dataset, args.train_batch, 
                                                        args.num_workers, loader = labeled_loader, device = device)
            x_ul, xw_ul, xs_wl, _, unlabeled_loader = get_next_batch(unlabeled_dataset, args.train_batch, 
                                                        args.num_workers, loader = unlabeled_loader, device = device)

            optimiser.zero_grad()

            labeled_outputs, _ = model(x_l)
            # Class predictions, final layer activations as feature vector
            unlabeled_outputs, unlabeled_features = model(x_ul)
            pseudo_label = torch.softmax(unlabeled_outputs.detach(), dim=-1)
            max_probs, predicted_cls = torch.max(pseudo_label, dim = -1)
            # Make all outputs lesser than threshold to -1, so that they can be filtered out
            # later
            predicted_cls[max_probs.lt(args.threshold)] = -1


            unlabeled_weakly_augmented_outputs, unlabeled_weakly_augmented_features = model(xw_ul)
            unlabeled_strongly_augmented_outputs, _ = model(xs_wl)

            supervised_loss = ce_loss_obj(labeled_outputs, y_l)
            consistency_loss = get_consistency_loss(unlabeled_weakly_augmented_outputs, unlabeled_strongly_augmented_outputs, args.threshold)
            contrastive_loss = contrastive_loss_obj(unlabeled_features, unlabeled_weakly_augmented_features, predicted_cls)

            loss = supervised_loss + args.fixmatch_alpha * consistency_loss + args.contrastive_alpha * contrastive_loss
            loss.backward()
            optimiser.step()

            if PLOT_AUGMENTATION:
                writer.add_images('Unlabelled', unnormalise(x_ul, args.mean, args.std), epoch * args.iter_per_epoch + i)
                writer.add_images('Weakly Augmented', unnormalise(xw_ul, args.mean, args.std), epoch * args.iter_per_epoch + i)
                writer.add_images('Strongly Augmented', unnormalise(xs_wl, args.mean, args.std), epoch * args.iter_per_epoch + i)

            writer.add_scalar('train/supervised_loss', supervised_loss.item(), epoch * args.iter_per_epoch + i)
            writer.add_scalar('train/consistency_loss', consistency_loss.item(), epoch * args.iter_per_epoch + i)
            writer.add_scalar('train/contrastive_loss', contrastive_loss.item(), epoch * args.iter_per_epoch + i)
            writer.add_scalar('train/total_loss', loss.item(), epoch * args.iter_per_epoch + i)
            writer.add_scalar('train/lr', scheduler.get_last_lr()[0], epoch * args.iter_per_epoch + i)

        validation_loss, validation_accuracy = test(model, val_loader, ce_loss_obj, device)
        writer.add_scalar('loss/validation', validation_loss, epoch)
        writer.add_scalar('accuracy/validation', validation_accuracy, epoch)

        scheduler.step()

        if epoch == 0 or (epoch + 1) % args.checkpoint_freq == 0:
            save_checkpoint(model, epoch, './checkpoints/{}/checkpoint_{}.pth'.format(current_time,epoch + 1),
                            optimiser = optimiser, params=get_params(args))

    test_loss, test_accuracy = test(model, test_loader, ce_loss_obj, device)

    writer.add_hparams( get_params(args),
                        {
                            'final_validation_loss': validation_loss,
                            'final_validation_accuracy': validation_accuracy,
                        })
    writer.add_scalar('loss/test', test_loss, epoch)
    writer.add_scalar('accuracy/test', test_accuracy, epoch)

    writer.close()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Custom SSL implementation using pytorch (FixMatch + SimCLRv2)")

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
                        help='Batchsize for train data')
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


     # Training Control parameters
    parser.add_argument('--total-iter', default=128*375, type=int,
                        help='total number of iterations to run')
    parser.add_argument('--iter-per-epoch', default=375, type=int,
                        help="Number of iterations to run per epoch")
    parser.add_argument("--checkpoint-freq", type=int, default=20, 
                        help="Save checkpoint after these many epochs")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--model-path", type=str)

    # FixMatch parameters
    parser.add_argument('--threshold', type=float, default=0.95,
                        help='Confidence Threshold for pseudo labeling')
    parser.add_argument("--fixmatch-alpha", default=1, type=float, 
                        help="Weight of fixmatch loss") 
    parser.add_argument("--contrastive-alpha", default=1, type=float, 
                        help="Weight of contrastive loss") 

    # Logging parameters
    parser.add_argument("--dataout", type=str, default="./runs/{}".format(current_time),
                        help="Path to save log files")

    args = parser.parse_args()

    if args.resume and args.model_path is None:
        parser.error("--resume requires --model-path."
                    "If resuming from previous checkpoint, provide checkpoint path")

    main(args)