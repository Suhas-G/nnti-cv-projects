
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

class VATLoss(nn.Module):

    def __init__(self, args):
        super(VATLoss, self).__init__()
        self.xi = args.vat_xi
        self.eps = args.vat_eps
        self.vat_iter = args.vat_iter

    def l2_norm(self, arr):
        # Assuming x is of shape, (batch_size, channels, height, width)
        return arr / (torch.linalg.norm(arr.view(arr.size()[0], -1, 1, 1), dim=1, ord=2, keepdim=True) + 1e-7)

    def forward(self, model, x):
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
        pred_adv = F.log_softmax(model(x + r_adv), dim = 1)

        return F.kl_div(pred_adv, pred, reduction='batchmean', log_target=True)