import torch
import torch.nn as nn
import torch.nn.functional as F

def kl_divergence(alpha, num_classes, device):
    ones = torch.ones([1, num_classes], dtype=torch.float32, device=device)
    sum_alpha = torch.sum(alpha, dim=1, keepdim=True)
    first_term = (
            torch.lgamma(sum_alpha)
            - torch.lgamma(alpha).sum(dim=1, keepdim=True)
            + torch.lgamma(ones).sum(dim=1, keepdim=True)
            - torch.lgamma(ones.sum(dim=1, keepdim=True))
    )
    second_term = (
        (alpha - ones)
        .mul(torch.digamma(alpha) - torch.digamma(sum_alpha))
        .sum(dim=1, keepdim=True)
    )
    kl = first_term + second_term
    return kl

def loglikelihood_loss(y, alpha, device):
    y = y.to(device)
    alpha = alpha.to(device)
    S = torch.sum(alpha, dim=1, keepdim=True)
    loglikelihood_err = torch.sum((y - (alpha / S)) ** 2, dim=1, keepdim=True)
    loglikelihood_var = torch.sum(
        alpha * (S - alpha) / (S * S * (S + 1)), dim=1, keepdim=True
    )
    loglikelihood = loglikelihood_err + loglikelihood_var
    return loglikelihood

def mse_loss(y, alpha, epoch_num, num_classes, annealing_step, device=None, useKL=True):
    y = y.to(device)
    alpha = alpha.to(device)
    loglikelihood = loglikelihood_loss(y, alpha, device=device)
    if not useKL:
        return loglikelihood
    annealing_coef = torch.min(
        torch.tensor(1.0, dtype=torch.float32),
        torch.tensor(epoch_num / annealing_step, dtype=torch.float32),)
    kl_alpha = (alpha - 1) * (1 - y) + 1
    kl_div = annealing_coef * kl_divergence(kl_alpha, num_classes, device=device)
    return loglikelihood + kl_div

def edl_loss(func, y, alpha, epoch_num, num_classes, annealing_step, device, useKL=True):
    y = y.to(device)
    alpha = alpha.to(device)
    S = torch.sum(alpha, dim=1, keepdim=True)
    A = torch.sum(y * (func(S) - func(alpha)), dim=1, keepdim=True)
    if not useKL:
        return A
    annealing_coef = torch.min(
        torch.tensor(1.0, dtype=torch.float32),
        torch.tensor(epoch_num / annealing_step, dtype=torch.float32),)
    kl_alpha = (alpha - 1) * (1 - y) + 1
    kl_div = annealing_coef * kl_divergence(kl_alpha, num_classes, device=device)
    return A + kl_div

def edl_mse_loss(alpha, target, epoch_num, num_classes, annealing_step, device):
    loss = mse_loss(target, alpha, epoch_num, num_classes, annealing_step, device=device)
    return torch.mean(loss)

def edl_log_loss(alpha, target, epoch_num, num_classes, annealing_step, device):
    loss = edl_loss(torch.log, target, alpha, epoch_num, num_classes, annealing_step, device)
    return torch.mean(loss)

def edl_digamma_loss(alpha, target, epoch_num, num_classes, annealing_step, device):
    loss = edl_loss(torch.digamma, target, alpha, epoch_num, num_classes, annealing_step, device)
    return torch.mean(loss)

def knn_distance(features, k=32):
    num_samples = features.size(0)
    if num_samples == 0:
        return 0 
    if num_samples <= k:
        k = num_samples
    distances = torch.cdist(features, features)
    _, indices = distances.topk(num_samples, dim=1, largest=False)
    return indices, distances

def knn_cosine_similarity(features, k=32):
    batch_size = features.size(0)
    if batch_size == 0:
        return 0
    if batch_size <= k:
        k = batch_size
    cosine_similarities = torch.matmul(features, features.t())
    values, indices = cosine_similarities.topk(k, dim=1, largest=True, sorted=True)
    return indices, values

def prototype_alignment_loss(num_classes, num_views, batch_size, prototypes, labels, device):
    loss = 0
    one_hot = F.one_hot(labels, num_classes).float().to(device)
    for v in range(num_views):
        view_prototypes = prototypes[v].to(device)
        selected_prototypes = view_prototypes[labels]
        cos_sim = F.cosine_similarity(selected_prototypes, one_hot, dim=1)
        proto_loss = 1 - cos_sim
        loss += proto_loss.sum()*0.1
    return loss / (batch_size * num_views)
    

def prototype_neighbor_loss(num_classes, num_views, batch_size, features, prototypes, Y, device, k=32, margin=1.0):
    loss = 0.0    
    for v in range(num_views):
        view_features = features[v].to(device)
        view_prototypes = prototypes[v].to(device)
        for c in range(num_classes):
            class_embeddings = view_features[Y == c]
            if knn_cosine_similarity(class_embeddings, k) == 0:
                continue
            else:
                indices, similarities = knn_cosine_similarity(class_embeddings, k)
                prototype = view_prototypes[c]
                knn_embeddings = view_features[indices]
                cosine_sim = F.cosine_similarity(prototype.unsqueeze(0), knn_embeddings)
                loss += (1 - cosine_sim).sum()
            prototype = view_prototypes[c]
            knn_embeddings = view_features
            cosine_sim = F.cosine_similarity(prototype.unsqueeze(0), knn_embeddings)
            loss += (1 - cosine_sim).sum()*0.1
        loss /= (num_classes * k)
    return loss

def get_prototypes(labels, class_prototypes):
    pos_proto = class_prototypes[labels]
    num_classes = class_prototypes.shape[0]
    cosine_sim = F.cosine_similarity(pos_proto.unsqueeze(1), class_prototypes.unsqueeze(0), dim=2)
    neg_indices = cosine_sim.argmin(dim=1)
    neg_proto = class_prototypes[neg_indices]
    return pos_proto, neg_proto

def triplet_loss_knn(num_classes, num_views, batch_size, features, prototypes, Y, device, k=32, margin=1.0):
    loss = 0
    for v in range(num_views):
        loss_v = 0
        view_features = features[v]
        view_prototypes = prototypes[v]
        for c in range(num_classes):
            pos_proto, neg_proto = get_prototypes(c , view_prototypes)
            pos_proto = pos_proto.to(device)
            neg_proto = neg_proto.to(device)
            class_embeddings = view_features[Y == c]
            if knn_cosine_similarity(class_embeddings, k) == 0:
                print('continue')
                continue
            else:
                indices, similarities = knn_cosine_similarity(class_embeddings, k)
                knn_embeddings = view_features[indices]
                knn_embeddings = knn_embeddings.to(device)
                pos_proto = pos_proto.unsqueeze(0) if pos_proto.dim() == 1 else pos_proto
                neg_proto = neg_proto.unsqueeze(0) if neg_proto.dim() == 1 else neg_proto
                pos_sim = F.cosine_similarity(view_features, pos_proto.unsqueeze(0))
                neg_sim = F.cosine_similarity(view_features, neg_proto.unsqueeze(0))
                loss_v += F.relu(margin - pos_sim + neg_sim).mean()
        loss += loss_v
    loss /= (num_classes * k)
    loss = loss.mean()*0.1
    return loss

def get_dc_loss(evidences, device):
    num_views = len(evidences)
    batch_size, num_classes = evidences[0].shape[0], evidences[0].shape[1]
    p = torch.zeros((num_views, batch_size, num_classes)).to(device)
    u = torch.zeros((num_views, batch_size)).to(device)
    for v in range(num_views):
        alpha = evidences[v] + 1
        S = torch.sum(alpha, dim=1, keepdim=True)
        p[v] = alpha / S
        u[v] = torch.squeeze(num_classes / S)
    dc_sum = 0
    for i in range(num_views):
        pd = torch.sum(torch.abs(p - p[i]) / 2, dim=2) / (num_views - 1)
        cc = (1 - u[i]) * (1 - u)
        dc = pd * cc
        dc_sum = dc_sum + torch.sum(dc, dim=0)
    dc_sum = torch.mean(dc_sum)
    return dc_sum

def EDL_loss(evidences, evidence_a, target, epoch_num, num_classes, annealing_step, gamma, device):
    target = F.one_hot(target, num_classes)
    alpha_a = evidence_a + 1
    loss_acc = edl_digamma_loss(alpha_a, target, epoch_num, num_classes, annealing_step, device)
    loss_acc = 0
    for v in range(len(evidences)):
        alpha = evidences[v] + 1
        loss_acc += edl_digamma_loss(alpha, target, epoch_num, num_classes, annealing_step, device)
    loss_acc = loss_acc / (len(evidences) + 1)
    loss = loss_acc + gamma * get_dc_loss(evidences, device)
    return loss

