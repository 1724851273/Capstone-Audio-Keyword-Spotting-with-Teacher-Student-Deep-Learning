import torch
import torch.nn as nn
import numpy as np
import random


class DML(nn.Module):

    def __init__(self):
        super(DML, self).__init__()
        self.project = nn.Sequential(
           
            nn.Linear(1024,128),
            nn.Tanh()
        )

        self.bias = nn.Parameter(torch.Tensor([0]))
        self.out = nn.Sigmoid()


    def forward_one(self, x):
        x = self.project(x)
        x = x.view(x.size(0), -1)    
        return x

    def forward(self, x1, x2):
        out1 = torch.squeeze(self.forward_one(x1))
        out2 = torch.squeeze(self.forward_one(x2))
        out = torch.diag(torch.matmul(out1, out2.T))
        out = self.out(out)# + self.bias)
        return out

class MaskedXentMarginLoss:
    def __init__(self, zero_weight=1.0, one_weight=1.0, margin=0.7, eps=1e-7, **kwargs):
        self.margin = margin
        self.zero_weight = zero_weight
        self.one_weight = one_weight
        self.eps = eps
        
    def __call__(self, pred, labels):
        pred = torch.clamp(pred, min=self.eps, max=1-self.eps)
        loss_mat = -self.one_weight * labels * torch.log(pred) * (pred <= self.margin).float()
        loss_mat -= self.zero_weight * (1 - labels) * torch.log(1-pred) * (pred >= 1 - self.margin).float()
        mask = ((labels == 0) + (labels == 1))
        loss = loss_mat[mask].sum()
        return loss
    
    
def get_batch(data, targets, speakers, batch_size=32):
    
    
    anchor = np.zeros((batch_size, data.shape[0]))
    friend = np.zeros((batch_size, data.shape[0]))
    foe = np.zeros((batch_size, data.shape[0]))
    
    for k in range(batch_size):
        while True:
            c1, c2 = random.sample(set(targets), 2)
            if not (len(c1)==0 or len(c2)==0 or c1 in c2 or c2 in c1):
                friend_clas_indices = np.where(targets == c1)[0]
                foe_clas_indices = np.where(targets == c2)[0]
                if not (len(friend_clas_indices)<2):
                    anchor_inx, friend_inx = random.sample(list(friend_clas_indices), 2)
                    if not(speakers[anchor_inx] == speakers[friend_inx]):
                        break
        foe_inx = random.sample(list(foe_clas_indices),1) [0]
        anchor[k, :] = data[:, anchor_inx]
        friend[k, :] = data[:, friend_inx]
        foe[k, :] = data[:, foe_inx]
        
    friend_label = np.ones(batch_size)
    foe_label = np.zeros(batch_size)
    
    batch_1 = torch.from_numpy(np.concatenate((anchor, anchor), axis = 0)).cuda()
    batch_2 = torch.from_numpy(np.concatenate((friend, foe),  axis = 0)).cuda()
    label = torch.from_numpy(np.concatenate((friend_label, foe_label))).cuda()

    return batch_1, batch_2, label