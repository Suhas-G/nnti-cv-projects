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

def save_checkpoint(model, epoch, filename, optimiser=None, save_arch=False, params=None):
    print('Saving checkpoint to {}'.format(filename))
    attributes = {
        'epoch': epoch,
        'state_dict': model.state_dict(),
    }

    if optimiser is not None:
        attributes['optimiser'] = optimiser.state_dict()

    if save_arch:
        attributes['arch'] = model

    if params is not None:
        attributes['params'] = params

    path = Path(filename)
    if not path.parent.exists():
        path.parent.mkdir(parents=True)
    try:
        torch.save(attributes, filename)
    except TypeError:
        if 'arch' in attributes:
            print('Model architecture will be ignored because the architecture includes non-pickable objects.')
            del attributes['arch']
            torch.save(attributes, filename)


def load_checkpoint(path, model=None, return_optimiser=False, return_params=False):
    print('Loading checkpoint from {}'.format(path))
    resume = torch.load(path)

    rets = dict()

    rets['epoch'] = resume['epoch']
    if model is not None:
        if ('module' in list(resume['state_dict'].keys())[0]) \
                and not (isinstance(model, torch.nn.DataParallel)):
            new_state_dict = OrderedDict()
            for k, v in resume['state_dict'].items():
                new_state_dict[k.replace('module.', '')] = v  # remove DataParallel wrapping

            model.load_state_dict(new_state_dict)
        else:
            model.load_state_dict(resume['state_dict'])

        rets['model'] = model

    if return_optimiser:
        rets['optimiser'] = resume['optimiser']
    if return_params:
        rets['params'] = resume['params']

    return rets


def load_model(path, model=None, is_inference=True):
    resume = torch.load(path)
    if model is None:
        model = resume['arch']
    model.load_state_dict(resume['state_dict'])
    if is_inference:
        model.eval()
    return model