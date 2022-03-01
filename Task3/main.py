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
from torch.utils.data import DataLoader, Subset
from torch.utils.tensorboard import SummaryWriter


torch.manual_seed(42)
random.seed(42)
np.random.seed(42)

from dataloader import get_cifar10, get_cifar100
from model.wrn import WideResNet
from utils import accuracy, load_checkpoint, save_checkpoint
from loss import ContrastiveLoss, get_consistency_loss

current_time = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M")

def create_validation_dataset(unlabelled_train, num_classes, size=1000):
    class_size = size // num_classes
    validation_idx = []
    for c in range(num_classes):
        c_targets = np.where(unlabelled_train.targets == c)[0]
        assert len(c_targets) >= class_size
        idx = np.random.choice(c_targets, class_size, replace=False)
        validation_idx.extend(idx)
    
    return Subset(unlabelled_train, validation_idx)

def get_datasets(args):
    if args.dataset == "cifar10":
        args.num_classes = 10
        labeled_dataset, unlabeled_dataset, test_dataset = get_cifar10(args, 
                                                                args.datapath)
    if args.dataset == "cifar100":
        args.num_classes = 100
        labeled_dataset, unlabeled_dataset, test_dataset = get_cifar100(args, 
                                                                args.datapath)

    validation_dataset = create_validation_dataset(unlabeled_dataset, args.num_classes, size=10000)

    return labeled_dataset, unlabeled_dataset, validation_dataset, test_dataset

def get_next_batch(dataset, batch_size, num_workers, loader = None, device = torch.device('cpu'), shuffle=True):
    try:
        img, weakly_augmented, strongly_augmented, target = next(loader)
    except (StopIteration, TypeError):
        loader = iter(DataLoader(dataset, batch_size=batch_size, 
                                shuffle=shuffle, num_workers=num_workers))
        img, weakly_augmented, strongly_augmented, target = next(loader)

    img, weakly_augmented, strongly_augmented, target = (img.to(device), weakly_augmented.to(device), 
                                                         strongly_augmented.to(device), target.to(device))
    return img, weakly_augmented, strongly_augmented, target, loader


def get_params(args):
    return {
        'dataset': args.dataset,
        'num_labeled': args.num_labeled,
        'lr': args.lr,
        'train_batch_size': args.train_batch,
        'test_batch_size': args.test_batch,
        'total_iter': args.total_iter,
        'iter_per_epoch': args.iter_per_epoch,
        'threshold': args.threshold,
        'model_depth': args.model_depth,
        'model_width': args.model_width
    }


def main(args):
    labeled_dataset, unlabeled_dataset, validation_dataset, test_dataset = get_datasets(args)
    args.epoch = math.ceil(args.total_iter / args.iter_per_epoch)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('Device: ', device)

    writer = SummaryWriter(log_dir=args.dataout)

    labeled_loader = None
    unlabeled_loader = None
    test_loader = None
    val_loader = None

    model = WideResNet(args.model_depth, args.num_classes, widen_factor=args.model_width)
    start_epoch = 0
    if args.resume:
        rets = load_checkpoint(args.model_path, model = model, return_optimiser=True)
        start_epoch = rets['epoch'] + 1
    model       = model.to(device)
    ce_loss_obj = nn.CrossEntropyLoss()
    contrastive_loss_obj = ContrastiveLoss(args.train_batch)

    optimiser = optim.Adam(model.parameters(), lr=args.lr)
    if args.resume:
        optimiser.load_state_dict(rets['optimiser'])


    for epoch in range(start_epoch, start_epoch + args.epoch):
        model.train()
        for i in range(args.iter_per_epoch):
            x_l, _, _, y_l, labeled_loader = get_next_batch(labeled_dataset, args.train_batch, 
                                                        args.num_workers, loader = labeled_loader, device = device)
            x_ul, xw_ul, xs_wl, _, unlabeled_loader = get_next_batch(unlabeled_dataset, unlabeled_loader, 
                                                        args.train_batch, args.num_workers, device = device)

            optimiser.zero_grad()

            labeled_outputs, _ = model(x_l)
            unlabeled_outputs, unlabeled_features = model(x_ul)
            pseudo_label = torch.softmax(unlabeled_outputs.detach(), dim=-1)
            max_probs, predicted_cls = torch.max(pseudo_label, dim = -1)
            predicted_cls[max_probs.lt(args.threshold)] = -1


            unlabeled_weakly_augmented_outputs, unlabeled_weakly_augmented_features = model(xw_ul)
            unlabeled_strongly_augmented_outputs, unlabeled_strongly_augmented_features = model(xs_wl)

            supervised_loss = ce_loss_obj(labeled_outputs, y_l)
            consistency_loss = get_consistency_loss(unlabeled_weakly_augmented_outputs, unlabeled_strongly_augmented_outputs, args.threshold)
            contrastive_loss = contrastive_loss_obj(unlabeled_features, unlabeled_weakly_augmented_features, predicted_cls)

            loss = supervised_loss + consistency_loss + contrastive_loss
            loss.backward()
            optimiser.step()

            writer.add_scalar('train/supervised_loss', supervised_loss.item(), epoch * args.iter_per_epoch + i)
            writer.add_scalar('train/consistency_loss', consistency_loss.item(), epoch * args.iter_per_epoch + i)
            writer.add_scalar('train/total_loss', loss.item(), epoch * args.iter_per_epoch + i)

        if epoch == 0 or (epoch + 1) % args.checkpoint_freq == 0:
            save_checkpoint(model, epoch, './checkpoints/{}/checkpoint_{}.pth'.format(current_time,epoch + 1),
                            optimiser = optimiser, params=get_params(args))
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


    # Logging parameters
    parser.add_argument("--dataout", type=str, default="./runs/{}".format(current_time),
                        help="Path to save log files")

    args = parser.parse_args()

    if args.resume and args.model_path is None:
        parser.error("--resume requires --model-path."
                    "If resuming from previous checkpoint, provide checkpoint path")

    main(args)