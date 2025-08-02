import torch
import torch.nn as nn
from collections import Counter
import torch.nn.functional as F

class EvidenceCollector(nn.Module):
    def __init__(self, dims, num_classes):
        super(EvidenceCollector, self).__init__()
        self.num_layers = len(dims) 
        self.net = nn.ModuleList() 
        for i in range(self.num_layers - 1):
            self.net.append(nn.Linear(dims[i], dims[i + 1])) 
            self.net.append(nn.ReLU()) 
            self.net.append(nn.Dropout(0.1))
        self.net.append(nn.Linear(dims[self.num_layers - 1], num_classes))
        self.net.append(nn.Softplus()) 
    def forward(self, x):
        h = self.net[0](x)
        for i in range(1, len(self.net)):
            h = self.net[i](h)
        return h

class SAPRMVL(nn.Module):
    def __init__(self, num_views, dims, num_classes, batch_size):
        super(SAPRMVL, self).__init__()
        self.num_views = num_views
        self.num_classes = num_classes
        self.dims = dims
        self.batch_size = batch_size
        self.FeatureCollectors = nn.ModuleList([
            FeatureCollector(dims[i], self.num_classes) for i in range(self.num_views)
        ])
        self.softplus = nn.Softplus()

    def forward(self, X, Y, noise_std=0.0):
        device = X[0].device
        features = {}
        evidences = {}
        prototypes = {}
        for v in range(self.num_views):
            feat = self.FeatureCollectors[v](X[v])
            if noise_std > 0:
                feat = feat + torch.randn_like(feat) * noise_std
            features[v] = feat
            evidences[v] = self.softplus(feat)
            prototypes[v] = self._compute_prototypes(feat, Y, self.num_classes, device)
        return features, prototypes, evidences

    @staticmethod
    def _compute_prototypes(features, labels, num_classes, device):
        B, D = features.size()
        prototypes = torch.zeros(num_classes, D, device=device)
        counts = torch.zeros(num_classes, device=device)
        prototypes = prototypes.index_add(0, labels, features)
        counts = counts.index_add(0, labels, torch.ones_like(labels, dtype=torch.float))
        counts = counts.clamp(min=1).unsqueeze(1)
        prototypes = prototypes / counts
        return prototypes

class FeatureCollector(nn.Module):
    def __init__(self, dims, num_classes):
        super(FeatureCollector, self).__init__()
        self.num_layers = len(dims)
        self.net = nn.ModuleList()
        if self.num_layers > 1:
            for i in range(self.num_layers - 1):
                self.net.append(nn.Linear(dims[i], dims[i + 1]))
                self.net.append(nn.ReLU())
                self.net.append(nn.Dropout(0.1))
            self.net.append(nn.Linear(dims[self.num_layers - 1], num_classes))
        else:
            self.net.append(nn.Linear(dims[self.num_layers - 1], num_classes))
    def forward(self, x):
        h = self.net[0](x)
        for i in range(1, len(self.net)):
            h = self.net[i](h)
        return h

def compute_kl_proto(P1, P2, eps=1e-8):
    P1 = F.softmax(P1, dim=1) + eps
    P2 = F.softmax(P2, dim=1) + eps
    kl = (P1 * (P1.log() - P2.log())).sum(dim=1)
    return kl.sum() 

def compute_similarity_weights(prototypes, evidences):
    M = len(prototypes)
    K = prototypes[0].shape[0] 
    similarity_weight = torch.nn.Parameter(torch.ones(M)) 
    uncertainty_weight = torch.nn.Parameter(torch.ones(M)) 
    similarities = torch.zeros(M, M)
    for i in range(M):
        for j in range(M):
            if i != j:
                similarities[i, j] = torch.exp(-compute_kl_proto(prototypes[i], prototypes[j]))
    w_S = similarities.sum(dim=1) / (similarities.sum() + 1e-8) 
    uncertainties = []
    for m in range(M):
        alpha = evidences[m] + 1 
        total_evidence = alpha.sum(dim=1) 
        u_m = K / (total_evidence + 1e-8) 
        u_mean = u_m.mean() 
        uncertainties.append(u_mean)
    uncertainties = torch.tensor(uncertainties, device=w_S.device)
    w_H = 1.0 / (uncertainties + 1e-8) 
    w = w_S * w_H
    w = w * torch.exp(similarity_weight) 
    w = w / (w.sum() + 1e-8) 
    w = w * torch.exp(uncertainty_weight) 
    w = w / (w.sum() + 1e-8) 
    return w

def fuse_evidence(evidence_list, weight, top_ratio=1):
    M = len(evidence_list)
    B, K = evidence_list[0].shape
    alpha_list = [e + 1 for e in evidence_list]
    weight = weight.clone().detach()
    topk_val, topk_idx = torch.topk(weight, int(M * top_ratio))
    valid_idx = topk_idx.tolist()
    final_alpha = torch.zeros(B, K)
    weight_sum = 0.0
    for m in valid_idx:
        w = weight[m]
        final_alpha += alpha_list[m] * w
        weight_sum += w
    final_alpha = final_alpha / (weight_sum + 1e-8)
    fused_evidence = final_alpha - 1.0
    return fused_evidence  # (B, K)
