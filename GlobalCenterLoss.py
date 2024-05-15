import torch
import numpy as np
import torch.nn as nn

class GlobalCenterTriplet(nn.Module):
    def __init__(self, margin=0.3, channel=2):
        super(GlobalCenterTriplet, self).__init__()
        self.margin, self.channel = margin, channel
        self.ranking_loss = nn.MarginRankingLoss(margin=margin)
        self.useful = None

    def forward(self, inputs, targets, center):
        n = inputs.size(0)
        center_need = center[targets]
        dist = torch.pow(inputs, 2).sum(dim=1, keepdim=True).expand(n, n)
        dist = dist + torch.pow(center_need, 2).sum(dim=1, keepdim=True).expand(n, n).t()
        dist.addmm_(1, -2, inputs, center_need.t())
        dist = dist.clamp(min=1e-12).sqrt()  # for numerical stability

        # For each anchor, find the hardest positive and negative
        mask = targets.expand(n, n).eq(targets.expand(n, n).t())
        dist_ap, dist_an = [], []
        for i in range(n):
            dist_ap.append(dist[i][i].unsqueeze(0))
            dist_an.append(dist[i][mask[i] == 0].min().unsqueeze(0))
            
        dist_ap = torch.cat(dist_ap)
        dist_an = torch.cat(dist_an)
        
        # Compute ranking hinge loss
        y = torch.ones_like(dist_an)
        loss = self.ranking_loss(dist_an, dist_ap, y)
        
        # compute accuracy
        correct = torch.ge(dist_an, dist_ap).sum().item()
        return loss, correct 