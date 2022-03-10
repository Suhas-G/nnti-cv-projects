
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable


class VATLoss(nn.Module):
    '''VAT Loss'''
    def __init__(self, args, writer = None, mean = None, std = None):
        super(VATLoss, self).__init__()
        self.xi = args.vat_xi
        self.eps = args.vat_eps
        self.vat_iter = args.vat_iter
        self.writer = writer
        self.mean = mean
        self.std = std
        self.count = 0

    def l2_norm(self, arr: torch.tensor):
        '''Calculate L2 norm of batch of tensors'''
        # Assuming x is of shape, (batch_size, channels, height, width)
        return arr / (torch.linalg.norm(arr.view(arr.size()[0], -1, 1, 1), dim=1, ord=2, keepdim=True) + 1e-7)

    def forward(self, model: torch.nn.Module, x: torch.tensor):
        # Random array chosen from Normal distribution with mean 0 and variance 1
        r = Variable(self.l2_norm(torch.normal(0, 1, size = x.size())).to(x.device), requires_grad = True)

        with torch.no_grad():
            pred = F.log_softmax(model(x), dim = 1)

        for _ in range(self.vat_iter):
            pred_adv = F.log_softmax(model(x + self.xi * r), dim = 1)
            adversarial_distance = F.kl_div(pred_adv, pred, reduction='batchmean', log_target=True)
            adversarial_distance.backward()
            r = Variable(self.l2_norm(r.grad), requires_grad = True)
            model.zero_grad()

        r_adv = r * self.eps
        x_adv = x + r_adv
        if self.writer is not None:
            self.plot_adversarial(x_adv.detach())
        pred_adv = F.log_softmax(model(x_adv), dim = 1)

        return F.kl_div(pred_adv, pred, reduction='batchmean', log_target=True)

    def plot_adversarial(self, images):
        images[:, 0] = (images[:, 0] * self.std[0]) + self.mean[0]
        images[:, 1] = (images[:, 1] * self.std[1]) + self.mean[1]
        images[:, 2] = (images[:, 2] * self.std[2]) + self.mean[2]
        self.writer.add_images('Adversarial Images', images, self.count)
        self.count += 1
