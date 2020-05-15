import torch
import torch.nn as nn
import torchvision.transforms as transforms

import numpy as np
import random

def random_aug(x):
    # gamma correction
    if random.random() <= 0.3:
        gamma = random.uniform(1.0, 1.5)
        x = gamma_correction(x, gamma)
    # random erasing with mean value
    mean_v = tuple(x.view(x.size(0), -1).mean(-1))
    re = transforms.RandomErasing(p=0.5, value=mean_v)
    x = re(x)
    # color channel shuffle
    if random.random() <= 0.3:
        l = [0,1,2]
        random.shuffle(l)
        x_c = torch.zeros_like(x)
        x_c[l] = x
        x = x_c
    # horizontal flip or vertical flip
    if random.random() <= 0.5:
        if random.random() <= 0.5:
            x = torch.flip(x, [1])
        else:
            x = torch.flip(x, [2])
    # rotate 90, 180 or 270 degree
    if random.random() <= 0.5:
        degree = [90, 180, 270]
        d = random.choice(degree)
        x = torch.rot90(x, d // 90, [1, 2])

    return x

class PseudoQeuryGenerator(object):
    def __init__(self, n_way, n_support, n_pseudo):
        super(PseudoQeuryGenerator, self).__init__()

        self.n_way            = n_way
        self.n_support        = n_support
        self.n_pseudo         = n_pseudo    # should be [100, 200].
        self.n_pseudo_per_way = self.n_pseudo // self.n_way

    def generate(self, support_set):

        mod_val = -1

        if self.n_support == 50:
            mod_val = 2 if self.n_pseudo == 100 else 4
            times = 1
        else:
            times = self.n_pseudo // (self.n_way * self.n_support)

        psedo_query_list = []
        for i in range(support_set.size(0)):
            if mod_val != -1 and i % 5 >= mod_val:
                continue

            for j in range(times):
                cur_x = support_set[i]
                cur_x = random_aug(cur_x)
                psedo_query_list.append(cur_x)

        psedo_query_set   = torch.stack(psedo_query_list)
        psedo_query_label = torch.from_numpy(np.repeat(range(self.n_way), self.n_pseudo_per_way))
        return psedo_query_set, psedo_query_label


def gamma_correction(x, gamma):
    minv = torch.min(x)
    x = x - minv
    maxv = torch.max(x)
    
    x = x / maxv
    x = x ** gamma
    x = x * maxv
    x = x - minv
    return x
