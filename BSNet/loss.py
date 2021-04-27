import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

class dice_bce_loss(nn.Module):
    def __init__(self, batch=True):
        super(dice_bce_loss, self).__init__()
        self.batch = batch
        self.bce_loss = nn.BCELoss()
        
    def soft_dice_coeff(self, y_true, y_pred):
        smooth = 0.0  # may change
        if self.batch:
            i = torch.sum(y_true)
            j = torch.sum(y_pred)
            intersection = torch.sum(y_true * y_pred)
        else:
            i = y_true.sum(1).sum(1).sum(1)
            j = y_pred.sum(1).sum(1).sum(1)
            intersection = (y_true * y_pred).sum(1).sum(1).sum(1)
        score = (2. * intersection + smooth) / (i + j + smooth)
        #score = (intersection + smooth) / (i + j - intersection + smooth)#iou
        return score.mean()

    def soft_dice_loss(self, y_true, y_pred):
        loss = 1 - self.soft_dice_coeff(y_true, y_pred)
        return loss
        
    def __call__(self, y_pred, y_true):
        a = self.bce_loss(y_pred, y_true)
        b = self.soft_dice_loss(y_true, y_pred)
        return a+b

class CrossEntropyLoss2d(nn.Module):
    def __init__(self, weight=None, size_average=False, ignore_index=255):
        super(CrossEntropyLoss2d, self).__init__()
        self.nll_loss = nn.NLLLoss2d(weight, size_average, ignore_index)

    def forward(self, inputs, targets):
        return self.nll_loss(F.log_softmax(inputs), targets)

def cross_entropy2d(input, target, weight=None, size_average=True):
    # input: (n, c, h, w), target: (n, h, w)
    n, c, h, w = input.size()
    # log_p: (n, c, h, w)
    log_p = F.log_softmax(input, dim=1).cuda()
    # log_p: (n*h*w, c)
    log_p = log_p.transpose(1, 2).transpose(2, 3).contiguous()
    log_p = log_p[target.view(n, h, w, 1).repeat(1, 1, 1, c) >= 0]
    log_p = log_p.view(-1, c)
    # target: (n*h*w,)
    # mask = (target != 255)
    # target = target[mask]
    loss = F.nll_loss(log_p, target, weight=weight, size_average=False, ignore_index=255).cuda()
    if size_average:
        loss /= (n*h*w)
    return loss

class FocalLoss2d(nn.Module):
    def __init__(self, gamma=2., weight=None, size_average=True):
        super(FocalLoss2d, self).__init__()
        self.gamma = gamma
        self.nll_loss = nn.NLLLoss2d(weight, size_average)

    def forward(self, inputs, targets):
        return self.nll_loss((1 - F.softmax(inputs)) ** self.gamma * F.log_softmax(inputs), targets)


class FocalLoss(nn.Module):
    """
        This criterion is a implemenation of Focal Loss, which is proposed in
        Focal Loss for Dense Object Detection.

            Loss(x, class) = - \alpha (1-softmax(x)[class])^gamma \log(softmax(x)[class])

        The losses are averaged across observations for each minibatch.
        Args:
            alpha(1D Tensor, Variable) : the scalar factor for this criterion
            gamma(float, double) : gamma > 0
            size_average(bool): size_average(bool): By default, the losses are averaged over observations for each minibatch.
                                However, if the field size_average is set to False, the losses are
                                instead summed for each minibatch.
    """

    def __init__(self, class_num=1, alpha=None, gamma=2, size_average=True):
        super(FocalLoss, self).__init__()
        if alpha is None:
            self.alpha = Variable(torch.ones(class_num+1))
        else:
            if isinstance(alpha, Variable):
                self.alpha = alpha
            else:
                self.alpha = Variable(alpha)
        self.gamma = gamma
        self.class_num = class_num
        self.size_average = size_average

    def forward(self, inputs, targets):  # variables
        b, c, h, w = inputs.size()
        class_mask = Variable(torch.zeros([b, c+1, h, w]).cuda())
        class_mask.scatter_(1, targets.long(), 1.)
        class_mask = class_mask[:, :-1, :, :]

        self.alpha = 0.75
        # if inputs.is_cuda and not self.alpha.is_cuda:
        #     self.alpha = self.alpha.cuda()
        # inputs[inputs<=0]=0.0001
        # inputs[inputs>=1]=0.9999
        # if P(gt=1)
        # focal_loss_1 = -alpha * np.power(1 - y, gamma) * np.log(y)
        # if P(gt=0)
        # focal_loss_2 = -(1 - alpha) * np.power(y, gamma) * np.log(1 - y)
        # batch_loss = -self.alpha * (torch.pow((1 - probs), self.gamma)) * log_p
        batch_loss = -targets*self.alpha*(torch.pow((1 - inputs), self.gamma))*torch.log(inputs+0.0001)\
                     - (1-targets)*(1-self.alpha)*(torch.pow(inputs, self.gamma))*torch.log(1.0001-inputs)
        batch_loss_1 = batch_loss.sum(dim=1).sum(1).sum(1) # (2,1,256,256) -> (2,256,256) -> (2,256) -> (2)
        if self.size_average:
            loss = batch_loss_1.mean()
        else:
            loss = batch_loss.sum()
        return loss

class loss_func(nn.Module):
    def __init__(self, batch=True):
        super(loss_func, self).__init__()
        self.batch = batch
        self.fl_loss = FocalLoss()
        self.dice_bce_loss = dice_bce_loss()

    def __call__(self, y_pred, y_true):
        focal_loss=self.fl_loss(y_pred,y_true)
        dice_loss=self.dice_bce_loss(y_true, y_pred)
        total_loss = focal_loss/1000 + dice_loss
        # print("focal loss"+str(focal_loss/1000)+"；dice loss"+str(dice_loss))
        return total_loss



class topo_loss(nn.Module):
    def __init__(self, batch=True):
        super(topo_loss, self).__init__()
        self.batch = batch
        self.mse_loss = nn.MSELoss()

    def topology_loss(self, y_true, y_pred):
        loss = self.mse_loss(y_pred, y_true)
        return loss

    def __call__(self, y_true, y_pred):
        c = self.topology_loss(y_true, y_pred)
        return c
