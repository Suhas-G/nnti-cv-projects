# Task 1 - Intructions

## Testing

The `test.py` file has code to test cifar10 and cifar100 datasets and also print the accuracy optionally.
Use `python test.py <arguments>` to run it. The arguments vary for each case and are given below.

To print the accuracy, add `--debug` to any of the following commands

1. CIFAR10, 250 labels, Threshold = 0.95

`python test.py --dataset cifar10 --num-labeled 250 --model-path ./checkpoints/cifar10_250_95/model.pth`

2. CIFAR10, 250 labels, Threshold = 0.75

`python test.py --dataset cifar10 --num-labeled 250 --model-path ./checkpoints/cifar10_250_75/model.pth`

3. CIFAR10, 250 labels, Threshold = 0.6

`python test.py --dataset cifar10 --num-labeled 250 --model-path ./checkpoints/cifar10_250_60/model.pth`

4. CIFAR10, 4000 labels, Threshold = 0.95

`python test.py --dataset cifar10 --num-labeled 4000 --model-path ./checkpoints/cifar10_4000_95/model.pth`

5. CIFAR10, 4000 labels, Threshold = 0.75

`python test.py --dataset cifar10 --num-labeled 4000 --model-path ./checkpoints/cifar10_4000_75/model.pth`

6. CIFAR10, 4000 labels, Threshold = 0.6

`python test.py --dataset cifar10 --num-labeled 4000 --model-path ./checkpoints/cifar10_4000_60/model.pth`

7. CIFAR100, 2500 labels, Threshold = 0.95

`python test.py --dataset cifar100 --num-labeled 2500 --model-path ./checkpoints/cifar100_2500_95/model.pth`

8. CIFAR100, 2500 labels, Threshold = 0.75

`python test.py --dataset cifar100 --num-labeled 2500 --model-path ./checkpoints/cifar100_2500_75/model.pth`

9. CIFAR100, 2500 labels, Threshold = 0.6

`python test.py --dataset cifar100 --num-labeled 2500 --model-path ./checkpoints/cifar100_2500_60/model.pth`

10. CIFAR100, 10000 labels, Threshold = 0.95

`python test.py --dataset cifar100 --num-labeled 10000 --model-path ./checkpoints/cifar100_10000_95/model.pth`

11. CIFAR100, 10000 labels, Threshold = 0.75

`python test.py --dataset cifar100 --num-labeled 10000 --model-path ./checkpoints/cifar100_10000_75/model.pth`

12. CIFAR100, 10000 labels, Threshold = 0.6

`python test.py --dataset cifar100 --num-labeled 10000 --model-path ./checkpoints/cifar100_10000_60/model.pth`




## Training

To reproduce the reported results please follow the following arguments.

1. CIFAR10, 250 labels, Threshold = 0.95

`python main.py --iter-per-epoch 512 --total-iter 20480 --checkpoint-freq 10 --train-batch 128 --test-batch 128 --dataset cifar10 --num-labeled 250 --threshold 0.95 --pseudo-loss-coeff 3 --pre-train 4`

2. CIFAR10, 250 labels, Threshold = 0.75

`python main.py --iter-per-epoch 512 --total-iter 20480 --checkpoint-freq 10 --train-batch 128 --test-batch 128 --dataset cifar10 --num-labeled 250 --threshold 0.75 --pseudo-loss-coeff 3 --pre-train 4`

3. CIFAR10, 250 labels, Threshold = 0.6

`python main.py --iter-per-epoch 512 --total-iter 20480 --checkpoint-freq 10 --train-batch 128 --test-batch 128 --dataset cifar10 --num-labeled 250 --threshold 0.6 --pseudo-loss-coeff 3 --pre-train 4`

4. CIFAR10, 4000 labels, Threshold = 0.95

`python main.py --iter-per-epoch 512 --total-iter 20480 --checkpoint-freq 10 --train-batch 128 --test-batch 128 --dataset cifar10 --num-labeled 4000 --threshold 0.95 --pseudo-loss-coeff 3 --pre-train 4`

5. CIFAR10, 4000 labels, Threshold = 0.75

`python main.py --iter-per-epoch 512 --total-iter 20480 --checkpoint-freq 10 --train-batch 128 --test-batch 128 --dataset cifar10 --num-labeled 4000 --threshold 0.75 --pseudo-loss-coeff 3 --pre-train 4`

6. CIFAR10, 4000 labels, Threshold = 0.6

`python main.py --iter-per-epoch 512 --total-iter 20480 --checkpoint-freq 10 --train-batch 128 --test-batch 128 --dataset cifar10 --num-labeled 4000 --threshold 0.6 --pseudo-loss-coeff 3 --pre-train 4`

7. CIFAR100, 2500 labels, Threshold = 0.95

`python main.py --iter-per-epoch 512 --total-iter 20480 --checkpoint-freq 10 --train-batch 128 --test-batch 128 --dataset cifar100 --num-labeled 2500 --threshold 0.95 --pseudo-loss-coeff 3 --pre-train 4`

8. CIFAR100, 2500 labels, Threshold = 0.75

`python main.py --iter-per-epoch 512 --total-iter 20480 --checkpoint-freq 10 --train-batch 128 --test-batch 128 --dataset cifar100 --num-labeled 2500 --threshold 0.75 --pseudo-loss-coeff 3 --pre-train 4`

9. CIFAR100, 2500 labels, Threshold = 0.6

`python main.py --iter-per-epoch 512 --total-iter 20480 --checkpoint-freq 10 --train-batch 128 --test-batch 128 --dataset cifar100 --num-labeled 2500 --threshold 0.6 --pseudo-loss-coeff 3 --pre-train 4`

10. CIFAR100, 10000 labels, Threshold = 0.95

`python main.py --iter-per-epoch 512 --total-iter 20480 --checkpoint-freq 10 --train-batch 128 --test-batch 128 --dataset cifar100 --num-labeled 10000 --threshold 0.95 --pseudo-loss-coeff 3 --pre-train 4`

11. CIFAR100, 10000 labels, Threshold = 0.75

`python main.py --iter-per-epoch 512 --total-iter 20480 --checkpoint-freq 10 --train-batch 128 --test-batch 128 --dataset cifar100 --num-labeled 10000 --threshold 0.75 --pseudo-loss-coeff 3 --pre-train 4`

12. CIFAR100, 10000 labels, Threshold = 0.6

`python main.py --iter-per-epoch 512 --total-iter 20480 --checkpoint-freq 10 --train-batch 128 --test-batch 128 --dataset cifar100 --num-labeled 10000 --threshold 0.95 --pseudo-loss-coeff 3 --pre-train 4`

To run on condor, modify the `arguments` section of the provided [docker.sub](./docker.sub) file accordingly.