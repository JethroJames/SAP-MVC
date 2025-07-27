import os
import torch
import torch.optim as optim
import time
import argparse
from utils import dict_to_namespace
import yaml
import numpy as np
from torch.utils.data import DataLoader, Subset
import torch.nn.functional as F
from loss_function import get_loss
import torch.nn as nn
from data import HandWritten, PIE,  ALOI
from model import SAPRMVL, fuse_evidence, compute_similarity_weights
import utils
from loss_function import prototype_alignment_loss,prototype_neighbor_loss,triplet_loss_knn, EDL_loss, fusion_loss, compute_saprmvl_loss

parser = argparse.ArgumentParser() 
#parser.add_argument('--config_file', type=str, default='configs/HandWritten.yaml') 
parser.add_argument('--config_file', type=str, default='configs/PIE.yaml')
#parser.add_argument('--config_file', type=str, default='configs/ALOI.yaml')
parser.add_argument('--seed', type=int, default=57) 
parser.add_argument('--gpu_id', type=int, default=-1) 
opt = parser.parse_args()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
with open(opt.config_file) as f:
    args = yaml.load(f, Loader=yaml.FullLoader)
args = dict_to_namespace(args)
path = args.dataset.dataPath

def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
seed = 57
set_seed(seed)

#dataset = HandWritten()
dataset = PIE()
#dataset = ALOI()
num_samples = len(dataset)
num_classes = args.base.num_classes
num_views = args.base.num_views
dims = dataset.dims
index = np.arange(num_samples)
np.random.shuffle(index)
train_index, test_index = index[:int(0.8 * num_samples)], index[int(0.8 * num_samples):]
train_loader = DataLoader(Subset(dataset, train_index), batch_size=args.base.batch_size, shuffle=True)
test_loader = DataLoader(Subset(dataset, test_index), batch_size=args.base.batch_size, shuffle=False)
#dataset.postprocessing(test_index, addNoise=True, sigma=0.5, ratio_noise=0.1, addConflict=True, ratio_conflict=0.4)
print(num_samples,num_classes,num_views,dims)


def train_proto(args, alpha=1, beta=0.005, gamma=1, margin=1):
    best_acc = 0.0
    best_model_state = None
    gg = 0.1
    model = SAPRMVL(args.base.num_views, dims, args.base.num_classes, args.base.batch_size)
    optimizer = optim.Adam(model.parameters(), lr=args.base.lr, weight_decay=1e-5)
    model.to(device)
    model.train() 
    loss_history = []
    for epoch in range(1, args.base.n_epochs + 1):
        print(f'==>{epoch}')
        total_loss = 0
        for X, Y, indexes in train_loader:
            for v in range(num_views):
                X[v] = X[v].to(device)
            Y = Y.to(device)
            features, prototypes, evidences = model(X, Y)
            weights = compute_similarity_weights(prototypes, evidences)
            evidence_a = fuse_evidence(evidences, weights)
            edl_loss = EDL_loss(evidences, evidence_a, Y, epoch, args.base.num_classes, args.base.annealing_step, gg, device)
            label_loss = prototype_alignment_loss(args.base.num_classes, args.base.num_views, args.base.batch_size,  prototypes, Y, device)
            neighbor_loss = prototype_neighbor_loss(args.base.num_classes, args.base.num_views, args.base.batch_size, features, prototypes, Y, device)
            triplet_loss = triplet_loss_knn(args.base.num_classes, args.base.num_views, args.base.batch_size, features, prototypes, Y, device, margin)
            l2_reg = 0
            for v in range(num_views):
                l2_reg += torch.sum(prototypes[v] ** 2)
            loss =  alpha * label_loss + beta * neighbor_loss + gamma * triplet_loss + edl_loss + 0.000005 * l2_reg
            optimizer.zero_grad()
            loss.backward(retain_graph=True)
            optimizer.step()
            total_loss += loss.item()
        avg_loss = total_loss / len(train_loader)
        loss_history.append(avg_loss)

        num_correct, num_sample = 0, 0
        with torch.no_grad():
            for X, Y, indexes in test_loader:
                for v in range(num_views):
                    X[v] = X[v].to(device)
                Y = Y.to(device)
                features, _ ,evidences = model(X, Y)
                evidence_a = evidences[0]
                for i in range(1, num_views):
                    evidence_a = (evidences[i] + evidence_a) / 2
                _, Y_pre = torch.max(evidence_a, dim=1)
                num_correct += (Y_pre == Y).sum().item()
                num_sample += Y.shape[0]
        acc = num_correct / num_sample
        print('====> acc: {:.4f}'.format(acc))
        if acc > best_acc:
            best_acc = acc
            best_model_state = model.state_dict()
            print(f"New best acc: {best_acc:.4f} (epoch {epoch})")
        print(f'Epoch {epoch}, Loss: {avg_loss}')
    model.eval()
    num_correct, num_sample = 0, 0
    with torch.no_grad():
        for X, Y, indexes in test_loader:
            for v in range(num_views):
                X[v] = X[v].to(device)
            Y = Y.to(device)
            features, _ ,evidences = model(X, Y)
            evidence_a = evidences[0]
            for i in range(1, num_views):
                evidence_a = (evidences[i] + evidence_a) / 2
            _, Y_pre = torch.max(evidence_a, dim=1)
            num_correct += (Y_pre == Y).sum().item()
            num_sample += Y.shape[0]

    acc = num_correct / num_sample
    print('====> acc: {:.4f}'.format(acc))
    if acc > best_acc:
        best_acc = acc
        best_model_state = model.state_dict()
        print(f"New best acc: {best_acc:.4f} (epoch {epoch})")
    print(f'best acc:{best_acc}')

if __name__ == '__main__':
    train_proto(args)
    

