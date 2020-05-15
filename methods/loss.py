import torch
from torch.autograd import Variable
import torch.nn as nn
import math
import numpy as np
import torch.nn.functional as F
from torch.nn.utils.weight_norm import WeightNorm
import utils

class FocalLoss(nn.Module):
    def __init__(self, gamma=0, eps=1e-7):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.eps = eps
        self.loss_fn = nn.CrossEntropyLoss()

    def forward(self, logits, y_query):
        logp = self.loss_fn(logits, y_query)
        p = torch.exp(-logp)
        loss = (1 - p) ** self.gamma * logp
        return loss.mean()

class PrototypeTripletLoss(nn.Module):
    def __init__(self, n_way, n_support):
        super(PrototypeTripletLoss, self).__init__()

        self.n_way     = n_way
        self.n_support = n_support
        self.loss_fn   = nn.TripletMarginLoss(margin=1.0, p=2)

    def forward(self, z_support, z_proto):
        z_anchor   = z_support.view(self.n_way * self.n_support, -1).repeat_interleave(self.n_way-1, dim=0)
        
        pid        = np.repeat(list(range(self.n_way)), (self.n_way-1) * self.n_support)
        z_positive = z_proto[pid]

        nid        = []
        for i in range(self.n_way):
            sl  = list(range(0, i)) + list(range(i+1, self.n_way))
            nid.extend(sl * self.n_support)
        z_negtive  = z_proto[nid]

        loss = self.loss_fn(z_anchor, z_positive, z_negtive)
        return loss

class LargeMarginCosineLoss(nn.Module):
    def __init__(self, s=30.0, m=0.35):
        super(LargeMarginCosineLoss, self).__init__()

        self.s       = s
        self.m       = m

    def forward(self, z_proto, z_query, y_query):
        norm_proto = F.normalize(z_proto, p=2, dim=1)
        norm_proto = norm_proto.transpose(0, 1)
        norm_query = F.normalize(z_query, p=2, dim=1)

        cos_theta = torch.mm(norm_query, norm_proto)
        cos_theta = cos_theta.clamp(-1,1) # for numerical stability

        phi     = cos_theta - self.m
        y_query = y_query.view(-1,1) #size=(B,1)
        index   = cos_theta.data * 0.0 #size=(B,Classnum)
        index.scatter_(1,y_query.data.view(-1,1),1)
        index = index.bool()

        output = cos_theta * 1.0
        output[index] = phi[index] #only change the correct predicted output
        output *= self.s # scale up in order to make softmax work, first introduced in normface
        return output


class Arcface(nn.Module):
    # implementation of additive margin softmax loss in https://arxiv.org/abs/1801.05599    
    def __init__(self, s=64.0, m=0.5):
        super(Arcface, self).__init__()

        self.s       = s
        self.m       = m

        self.cos_m = math.cos(m)
        self.sin_m = math.sin(m)

        self.mm = self.sin_m * m  # issue 1
        self.threshold = math.cos(math.pi - m)

    def forward(self, z_proto, z_query, y_query):
        # weights norm
        nB = z_query.size(0)
        norm_proto = F.normalize(z_proto, p=2, dim=1)
        norm_proto = norm_proto.transpose(0, 1)
        norm_feats = F.normalize(z_query, p=2, dim=1)
        # cos(theta+m)
        cos_theta = torch.mm(norm_feats,norm_proto)

        cos_theta = cos_theta.clamp(-1,1) # for numerical stability
        cos_theta_2 = torch.pow(cos_theta, 2)
        sin_theta_2 = 1 - cos_theta_2
        sin_theta = torch.sqrt(sin_theta_2)
        cos_theta_m = (cos_theta * self.cos_m - sin_theta * self.sin_m)

        cond_v = cos_theta - self.threshold
        cond_mask = cond_v <= 0
        keep_val = (cos_theta - self.mm) # when theta not in [0,pi], use cosface instead
        cos_theta_m[cond_mask] = keep_val[cond_mask]
        output = cos_theta * 1.0 # a little bit hacky way to prevent in_place operation on cos_theta
        idx_ = torch.arange(0, nB, dtype=torch.long)
        output[idx_, y_query] = cos_theta_m[idx_, y_query]
        output *= self.s # scale up in order to make softmax work, first introduced in normface

        return output

#-----------------------------------------------------------------------------------------------------
class Cosface_Linear(nn.Module):
    # implementation of additive margin softmax loss in https://arxiv.org/abs/1801.05599    
    def __init__(self,embedding_size=512,classnum=5):
        super(Cosface_Linear, self).__init__()
        self.classnum = classnum
        self.kernel = nn.Parameter(torch.Tensor(embedding_size,classnum))
        # initial kernel
        self.kernel.data.uniform_(-1, 1).renorm_(2,1,1e-5).mul_(1e5)
        self.m = 0.35 # additive margin recommended by the paper
        self.s = 30. # see normface https://arxiv.org/abs/1704.06369

    def forward(self,embbedings,label):
        kernel_norm = F.normalize(self.kernel, p=2, dim=0)
        embbedings_norm = F.normalize(embbedings, p=2, dim=1)

        cos_theta = torch.mm(embbedings_norm, kernel_norm)
        cos_theta = cos_theta.clamp(-1,1) # for numerical stability
        phi = cos_theta - self.m
        label = label.view(-1,1) #size=(B,1)
        index = cos_theta.data * 0.0 #size=(B,Classnum)
        index.scatter_(1,label.data.view(-1,1),1)
        index = index.bool()
        output = cos_theta * 1.0
        output[index] = phi[index] #only change the correct predicted output
        output *= self.s # scale up in order to make softmax work, first introduced in normface
        return output

    def correct(self,embbedings):
        kernel_norm = F.normalize(self.kernel, p=2, dim=0)
        embbedings_norm = F.normalize(embbedings, p=2, dim=1)

        cos_theta = torch.mm(embbedings_norm, kernel_norm)
        cos_theta = cos_theta.clamp(-1,1) # for numerical stability

        return cos_theta


class Arcface_Linear(nn.Module):
    # implementation of additive margin softmax loss in https://arxiv.org/abs/1801.05599    
    def __init__(self, embedding_size=512, classnum=5,  s=64., m=0.5):
        super(Arcface_Linear, self).__init__()
        self.classnum = classnum
        self.kernel = nn.Parameter(torch.Tensor(embedding_size,classnum))
        # initial kernel
        self.kernel.data.uniform_(-1, 1).renorm_(2,1,1e-5).mul_(1e5)
        self.m = m # the margin value, default is 0.5
        self.s = s # scalar value default is 64, see normface https://arxiv.org/abs/1704.06369
        self.cos_m = math.cos(m)
        self.sin_m = math.sin(m)
        self.mm = self.sin_m * m  # issue 1
        self.threshold = math.cos(math.pi - m)
    def forward(self, embbedings, label):
        # weights norm
        nB = len(embbedings)
        kernel_norm = F.normalize(self.kernel, p=2, dim=0)
        embbedings_norm = F.normalize(embbedings, p=2, dim=1)

        cos_theta = torch.mm(embbedings_norm,kernel_norm)
        cos_theta = cos_theta.clamp(-1,1) # for numerical stability
        cos_theta_2 = torch.pow(cos_theta, 2)
        sin_theta_2 = 1 - cos_theta_2
        sin_theta = torch.sqrt(sin_theta_2)
        cos_theta_m = (cos_theta * self.cos_m - sin_theta * self.sin_m)

        cond_v = cos_theta - self.threshold
        cond_mask = cond_v <= 0
        keep_val = (cos_theta - self.mm) # when theta not in [0,pi], use cosface instead
        cos_theta_m[cond_mask] = keep_val[cond_mask]
        output = cos_theta * 1.0 # a little bit hacky way to prevent in_place operation on cos_theta
        idx_ = torch.arange(0, nB, dtype=torch.long)
        output[idx_, label] = cos_theta_m[idx_, label]
        output *= self.s # scale up in order to make softmax work, first introduced in normface
        return output

    def correct(self,embbedings):
        kernel_norm = F.normalize(self.kernel, p=2, dim=0)
        embbedings_norm = F.normalize(embbedings, p=2, dim=1)

        cos_theta = torch.mm(embbedings_norm, kernel_norm)
        cos_theta = cos_theta.clamp(-1,1) # for numerical stability

        return cos_theta
