import torch
from collections import OrderedDict
from pathlib import Path

def accuracy(output, target, topk=(1,)):
    """
    Function taken from pytorch examples:
    https://github.com/pytorch/examples/blob/master/imagenet/main.py
    Computes the accuracy over the k top predictions for the specified
    values of k
    """
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res

def save_checkpoint(model: torch.nn.Module, epoch: int, filename: str, 
                   optimiser: torch.optim.Optimizer = None, params = None):
    print('Saving checkpoint to {}'.format(filename))
    data = {
        'epoch': epoch,
        'state_dict': model.state_dict(),
    }

    if optimiser is not None:
        data['optimiser'] = optimiser.state_dict()


    if params is not None:
        data['params'] = params

    path = Path(filename)
    if not path.parent.exists():
        path.parent.mkdir(parents=True)

    torch.save(data, filename)

def load_checkpoint(path: str, model: torch.nn.Module = None, 
                    return_optimiser: bool = False, return_params: bool = False):
    print('Loading checkpoint from {}'.format(path))
    resume = torch.load(path)
    rets = dict()

    rets['epoch'] = resume['epoch']
    if model is not None:
        model.load_state_dict(resume['state_dict'])

        rets['model'] = model

    if return_optimiser:
        rets['optimiser'] = resume['optimiser']
    if return_params:
        rets['params'] = resume['params']

    return rets