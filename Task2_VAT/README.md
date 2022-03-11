# Task 2 - Intructions

## Testing

The `test.py` file has code to test cifar10 and cifar100 datasets and also print the accuracy optionally.
Use `python test.py <arguments>` to run it. The arguments vary for each case and are given below.

To print the accuracy, add `--debug` to any of the following commands

1. CIFAR10, 250 labels

`python test.py --dataset cifar10 --num-labeled 250 --model-path ./checkpoints/cifar10_250/model.pth`

2. CIFAR10, 4000 labels

`python test.py --dataset cifar10 --num-labeled 4000 --model-path ./checkpoints/cifar10_4000/model.pth`

3. CIFAR100, 2500 labels

`python test.py --dataset cifar100 --num-labeled 2500 --model-path ./checkpoints/cifar100_2500/model.pth`

4. CIFAR100, 10000 labels

`python test.py --dataset cifar100 --num-labeled 10000 --model-path ./checkpoints/cifar100_10000/model.pth`


## Training

To reproduce the reported results please follow the following arguments.

1. CIFAR10, 250 labels

`python main.py --num-labeled 250 --total-iter 25600 --iter-per-epoch 512 --dataset cifar10 --train-labelled-batch 64 --train-unlabelled-batch 128 --vat-xi 1 --vat-eps 2 --vat-alpha 1 --vat-iter 1`

2. CIFAR10, 4000 labels

`python main.py --num-labeled 4000 --total-iter 25600 --iter-per-epoch 512 --dataset cifar10 --train-labelled-batch 64 --train-unlabelled-batch 128 --vat-xi 1 --vat-eps 2 --vat-alpha .5 --vat-iter 1`

3. CIFAR100, 2500 labels

`python main.py --num-labeled 2500 --total-iter 25600 --iter-per-epoch 512 --dataset cifar100 --train-labelled-batch 64 --train-unlabelled-batch 128 --vat-xi 1 --vat-eps 2.5 --vat-alpha .5 --vat-iter 1`

4. CIFAR100, 10000 labels

`python main.py --num-labeled 10000 --total-iter 25600 --iter-per-epoch 512 --dataset cifar100 --train-labelled-batch 64 --train-unlabelled-batch 128 --vat-xi 1 --vat-eps 2 --vat-alpha .5 --vat-iter 1`

To run on condor, modify the `arguments` section of the provided [docker.sub](./docker.sub) file accordingly.
