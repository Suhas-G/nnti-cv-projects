import numpy as np
import torch
import math

import torchvision.datasets as datasets
import torchvision.transforms as transforms
from PIL import Image

cifar10_mean    = [0.4914, 0.4822, 0.4465]
cifar10_std     = [0.2471, 0.2435, 0.2616]
cifar100_mean   = [0.5071, 0.4867, 0.4408]
cifar100_std    = [0.2675, 0.2565, 0.2761]

def x_u_split(args, labels):
    label_per_class = args.num_labeled // args.num_classes
    labels = np.array(labels)
    labeled_idx = []
    unlabeled_idx = np.array(range(len(labels)))
    for i in range(args.num_classes):
        idx = np.where(labels == i)[0]
        idx = np.random.choice(idx, label_per_class, False)
        labeled_idx.extend(idx)
    labeled_idx = np.array(labeled_idx)
    assert len(labeled_idx) == args.num_labeled

    unlabeled_idx = np.setdiff1d(unlabeled_idx, labeled_idx)

    if args.expand_labels or args.num_labeled < args.train_batch:
        num_expand_x = math.ceil(
            args.train_batch * args.iter_per_epoch / args.num_labeled)
        labeled_idx = np.hstack([labeled_idx for _ in range(num_expand_x)])
    np.random.shuffle(labeled_idx)
    return labeled_idx, unlabeled_idx

def get_cifar10(args, root):
    transform_labeled = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(size=32,
                              padding=int(32*0.125),
                              padding_mode='reflect'),
        transforms.ToTensor(),
        transforms.Normalize(mean=cifar10_mean, std=cifar10_std)
    ])

    transform_weak = transforms.Compose([
        transforms.RandomHorizontalFlip(p = 0.5),
        transforms.RandomAffine(0, translate=(0.125, 0.125)),
        transforms.ToTensor(),
        transforms.Normalize(mean=cifar10_mean, std=cifar10_std)
    ])

    transform_strong = transforms.Compose([
        transforms.RandAugment(),
        transforms.ToTensor(),
        transforms.Normalize(mean=cifar10_mean, std=cifar10_std)
    ])

    transform_val = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=cifar10_mean, std=cifar10_std)
    ])
    base_dataset = datasets.CIFAR10(root, train=True, download=True)

    train_labeled_idxs, train_unlabeled_idxs = x_u_split(
        args, base_dataset.targets)

    train_labeled_dataset = CIFAR10SSL(
        root, train_labeled_idxs, train=True,
        transform=transform_labeled)

    train_unlabeled_dataset = CIFAR10SSL(
        root, train_unlabeled_idxs, train=True,
        transform_weak = transform_weak,
        transform_strong = transform_strong)

    test_dataset = datasets.CIFAR10(
        root, train=False, transform=transform_val, download=False)

    return train_labeled_dataset, train_unlabeled_dataset, test_dataset


def get_cifar100(args, root):

    transform_labeled = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(size=32,
                              padding=int(32*0.125),
                              padding_mode='reflect'),
        transforms.ToTensor(),
        transforms.Normalize(mean=cifar100_mean, std=cifar100_std)])
    
    transform_weak = transforms.Compose([
        transforms.RandomHorizontalFlip(p = 0.5),
        transforms.RandomAffine(0, translate=(0.125, 0.125)),
        transforms.ToTensor(),
        transforms.Normalize(mean=cifar100_mean, std=cifar100_std)
    ])

    transform_strong = transforms.Compose([
        transforms.RandAugment(),
        transforms.ToTensor(),
        transforms.Normalize(mean=cifar100_mean, std=cifar100_std)
    ])

    transform_val = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=cifar100_mean, std=cifar100_std)])

    base_dataset = datasets.CIFAR100(
        root, train=True, download=True)

    train_labeled_idxs, train_unlabeled_idxs = x_u_split(
        args, base_dataset.targets)

    train_labeled_dataset = CIFAR100SSL(
        root, train_labeled_idxs, train=True,
        transform=transform_labeled)

    train_unlabeled_dataset = CIFAR100SSL(
        root, train_unlabeled_idxs, train=True,
        transform_weak = transform_weak,
        transform_strong = transform_strong)

    test_dataset = datasets.CIFAR100(
        root, train=False, transform=transform_val, download=False)

    return train_labeled_dataset, train_unlabeled_dataset, test_dataset

class CIFAR10SSL(datasets.CIFAR10):
    def __init__(self, root, indexs, train=True,
                 transform = None,
                 weak_transform=None, 
                 strong_transform=None,
                 target_transform=None,
                 download=False):
        super().__init__(root, train=train,
                         transform=transform,
                         target_transform=target_transform,
                         download=download)
        if indexs is not None:
            self.data = self.data[indexs]
            self.targets = np.array(self.targets)[indexs]
            self.weak_transform = weak_transform
            self.strong_transform = strong_transform

    def __getitem__(self, index):
        img, target = self.data[index], self.targets[index]
        img = Image.fromarray(img)

        weakly_augmented = None
        strongly_augmented = None

        if self.transform is not None:
            img = self.transform(img)

        if self.weak_transform is not None:
            weakly_augmented = self.weak_transform(img)

        if self.strong_transform is not None:
            strongly_augmented = self.strong_transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        target = torch.tensor(target)
        return torch.tensor(img), weakly_augmented, strongly_augmented, target.long()

class CIFAR100SSL(datasets.CIFAR100):
    def __init__(self, root, indexs, train=True,
                 transform = None,
                 weak_transform=None, 
                 strong_transform=None,
                 target_transform=None,
                 download=False):
        super().__init__(root, train=train,
                         transform = transform,
                         target_transform=target_transform,
                         download=download)
        if indexs is not None:
            self.data = self.data[indexs]
            self.targets = np.array(self.targets)[indexs]
            self.weak_transform = weak_transform
            self.strong_transform = strong_transform

    def __getitem__(self, index):
        img, target = self.data[index], self.targets[index]
        img = Image.fromarray(img)
        weakly_augmented = None
        strongly_augmented = None

        if self.transform is not None:
            img = self.transform(img)

        if self.weak_transform is not None:
            weakly_augmented = self.weak_transform(img)

        if self.strong_transform is not None:
            strongly_augmented = self.strong_transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        target = torch.tensor(target)
        return torch.tensor(img), weakly_augmented, strongly_augmented, target.long()

