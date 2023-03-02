import torch.optim.lr_scheduler
import pickle
import numpy as np
import torch.nn as nn
import torch
from sklearn.linear_model import LogisticRegression
from tqdm import tqdm
use_gpu = torch.cuda.is_available()
import sklearn
import sys
from torch.utils.data import Dataset
from torchvision.transforms import ToTensor
from torchvision import datasets
from torch.utils.data import DataLoader
import torch.optim as optim
import pickle as pkl
import random
SEED = 42
from slowfast.models.head_helper import TransformerBasicHead
random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
np.random.seed(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
from sklearn.metrics import confusion_matrix

from einops import rearrange, repeat
from einops.layers.torch import Rearrange
global nc
nc = 34
# helpers

def pair(t):
    return t if isinstance(t, tuple) else (t, t)

# classes

def random_feature_interpolation(selected_mean, query, num):
    alpha = np.random.rand((20,768))*0.05
    generated_feature = query + alpha*selected_mean
    return generated_feature

class CustomImageDataset(Dataset):
    def __init__(self, feature, annotation, transform=None, target_transform=None):

        self.feature = feature #.view(-1,34)
        #print(self.feature.shape)
        #sys-exit()
        self.annotations = annotation
    def __len__(self):
        return len(self.annotations)
    def __getitem__(self, idx):
        return self.feature[idx], self.annotations[idx]

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.head = TransformerBasicHead(dim_in=768,num_classes=20)
    def forward(self, x):
        return self.head(x)

def class_assignment(train_dataset, train_labels):
    cls_set = []*20
    cls_set_annotation = []*20
    for i, (data, label) in enumerate(zip(train_dataset, train_labels)):
        for j in range(20):
            if label[j] == 1:
                cls_set[j].append(data)
                cls_set_annotation[j].append(label)
    return cls_set, cls_set_annotation

def generate_train(train_dataset, train_labels):
    cls_set, cls_set_annotation = class_assignment(train_dataset, train_labels)
    base_means = []
    base_cov = []
    for key in range(20):
        feature = np.array(cls_set[key]) # N,20,768
        #print(feature.shape)
        mean = np.mean(feature, axis=0)
        #cov = np.cov(feature.T)
        base_means.append(mean)
        #base_cov.append(cov)
    sampled_data = []
    sampled_label = []
    count = 0
    np.set_printoptions(threshold=sys.maxsize)
    for i, (sample,label) in enumerate(zip(train_dataset,train_labels)):
        empty_set = []
        for j in range(20):
            if label[j] == 0:
                empty_set.append(j)
        if len(empty_set) >= 1:
            for k in range(3):
                selected = np.random.choice(empty_set,1)
                mean, cov, samples = random_feature_interpolation(sample, base_means[selected], 1)
                label[selected] = 1
                sampled_data.append(samples)
                sampled_label.append(label)
    sampled_data = train_dataset + sampled_data
    sampled_label = train_labels + sampled_label
    return sampled_data, sampled_label

if __name__ == '__main__':
    # ---- data loading
    n_runs = 10000

    f = open('/hkfs/work/workspace/scratch/fy2374-musclemap/models/model_set_1/25_09/ablation_module3_2/train_embeddding_set.pkl','rb')
    train_dataset = pkl.load(f)
    f.close()
    f = open('/hkfs/work/workspace/scratch/fy2374-musclemap/models/model_set_1/25_09/ablation_module3_2/val_seen_embeddding_set.pkl','rb')
    val_seen_dataset = pkl.load(f)
    f.close()
    f = open('/hkfs/work/workspace/scratch/fy2374-musclemap/models/model_set_1/25_09/ablation_module3_2/val_unseen_embeddding_set.pkl','rb')
    val_unseen_dataset = pkl.load(f)
    f.close()
    f = open('/hkfs/work/workspace/scratch/fy2374-musclemap/models/model_set_1/25_09/ablation_module3_2/test_seen_embeddding_set.pkl','rb')
    test_seen_dataset = pkl.load(f)
    f.close()
    f = open('/hkfs/work/workspace/scratch/fy2374-musclemap/models/model_set_1/25_09/ablation_module3_2/test_unseen_embeddding_set.pkl','rb')
    test_unseen_dataset= pkl.load(f)
    f.close()
    f = open('/hkfs/work/workspace/scratch/fy2374-musclemap/models/model_set_1/25_09/ablation_module3_2/train_label_set.pkl','rb')
    train_labels = pkl.load(f)
    f.close()
    f = open('/hkfs/work/workspace/scratch/fy2374-musclemap/models/model_set_1/25_09/ablation_module3_2/val_seen_label_set.pkl','rb')
    val_seen_labels = pkl.load(f)
    f.close()
    f = open('/hkfs/work/workspace/scratch/fy2374-musclemap/models/model_set_1/25_09/ablation_module3_2/val_unseen_label_set.pkl','rb')
    val_unseen_labels = pkl.load(f)
    f.close()
    f = open('/hkfs/work/workspace/scratch/fy2374-musclemap/models/model_set_1/25_09/ablation_module3_2/test_seen_label_set.pkl','rb')
    test_seen_labels = pkl.load(f)
    f.close()
    f = open('/hkfs/work/workspace/scratch/fy2374-musclemap/models/model_set_1/25_09/ablation_module3_2/test_unseen_label_set.pkl','rb')
    test_unseen_labels = pkl.load(f)
    f.close()
    train_dataset, train_labels = generate_train(train_dataset, train_labels)
    dataset_train = CustomImageDataset(np.squeeze(train_dataset), train_labels)
    dataset_val_seen = CustomImageDataset(np.squeeze(val_seen_dataset), val_seen_labels)
    dataset_test_seen = CustomImageDataset(np.squeeze(test_seen_dataset), test_seen_labels)
    dataset_val_unseen = CustomImageDataset(np.squeeze(val_unseen_dataset), val_unseen_labels)
    dataset_test_unseen = CustomImageDataset(np.squeeze(test_unseen_dataset), test_unseen_dataset)
    train_dataloader = DataLoader(dataset_train, batch_size=256, shuffle=True)
    infer_dataloader = DataLoader(dataset_train, batch_size=256, shuffle=False)
    test_seen_dataloader = DataLoader(dataset_test_seen, batch_size=64, shuffle=False)
    val_seen_dataloader = DataLoader(dataset_val_seen, batch_size=64, shuffle=False)
    test_unseen_dataloader = DataLoader(dataset_test_unseen, batch_size=64, shuffle=False)
    val_unseen_dataloader = DataLoader(dataset_val_unseen, batch_size=64, shuffle=False)
    model = Net()
    model=model.cuda()
    criterion = torch.nn.BCELoss(weight=None, size_average=None, reduce=None, reduction='mean')
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.0001, betas=(0.9, 0.999), eps=1e-08, weight_decay=0.01, amsgrad=False)

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=5,  eta_min=0, last_epoch=- 1, verbose=False)
    for epoch in range(1500):
        model.train()
        for step, (data,label) in enumerate(train_dataloader):
            optimizer.zero_grad()
            predicts,y = model(data.cuda(), data.cuda())
            loss = criterion(predicts, label.cuda()) #+ criterion2(predicts,y, torch.ones(y.size()[0]).cuda())
            loss.backward()
            optimizer.step()
            scheduler.step()
        print(epoch, 'loss', loss)
    model.eval()

    val_predict_seen = []
    val_label_seen = []
    for step, (data,label) in enumerate(val_seen_dataloader):
        with torch.no_grad():
            #data = torch.nn.functional.normalize(data, dim=-1)

            predicts,y = model(data.cuda(), data.cuda())
            val_predict_seen.append(predicts.cpu())
            val_label_seen.append(label.cpu())
    val_predict_seen = torch.cat(val_predict_seen, dim=0).cpu().numpy()
    val_label_seen = torch.cat(val_label_seen, dim=0).cpu().numpy()
    #val_predict = np.argmax(val_predict, axis=-1)
    #print(predicts)
    #acc = sklearn.metrics.top_k_accuracy_score(val_seen_labels, val_predict_seen, k=1)
    pred_list = val_predict_seen[:, ~(np.all(val_label_seen == 0, axis=0))]
    label_list = val_label_seen[:, ~(np.all(val_label_seen == 0, axis=0))]
    mAP =sklearn.metrics.average_precision_score(label_list,pred_list)
    print('calibrated mAP val seen:', mAP)
    #confusion = confusion_matrix(label_list, pred_list)
    #list_diag = np.diag(confusion)
    #list_raw_sum = np.sum(confusion, axis=1)
    # each_acc = list_diag / list_raw_sum
    val_predict_unseen = []
    val_label_unseen = []
    for step, (data,label) in enumerate(val_unseen_dataloader):
        with torch.no_grad():
            #data = torch.nn.functional.normalize(data, dim=-1)

            predicts,y = model(data.cuda(), data.cuda())
            val_predict_unseen.append(predicts.cpu())
            val_label_unseen.append(label.cpu())
    val_predict_unseen = torch.cat(val_predict_unseen, dim=0).cpu().numpy()
    val_label_unseen = torch.cat(val_label_unseen, dim=0).cpu().numpy()
    #val_predict = np.argmax(val_predict, axis=-1)
    #print(predicts)
    #acc = sklearn.metrics.top_k_accuracy_score(val_seen_labels, val_predict_seen, k=1)
    pred_list = val_predict_unseen[:, ~(np.all(val_label_unseen == 0, axis=0))]
    label_list = val_label_unseen[:, ~(np.all(val_label_unseen == 0, axis=0))]
    mAP =sklearn.metrics.average_precision_score(label_list,pred_list)
    print('calibrated mAP val unseen:', mAP)

    test_predict_seen = []
    test_label_seen = []
    for step, (data,label) in enumerate(test_seen_dataloader):
        with torch.no_grad():
            #data = torch.nn.functional.normalize(data, dim=-1)

            predicts,y = model(data.cuda(), data.cuda())
            test_predict_seen.append(predicts.cpu())
            test_label_seen.append(label.cpu())
    test_predict_seen = torch.cat(test_predict_seen, dim=0).cpu().numpy()
    test_label_seen = torch.cat(test_label_seen, dim=0).cpu().numpy()
    #val_predict = np.argmax(val_predict, axis=-1)
    #print(predicts)
    #acc = sklearn.metrics.top_k_accuracy_score(val_seen_labels, val_predict_seen, k=1)
    pred_list = test_predict_seen[:, ~(np.all(test_label_seen == 0, axis=0))]
    label_list = test_label_seen[:, ~(np.all(test_label_seen == 0, axis=0))]
    mAP =sklearn.metrics.average_precision_score(label_list,pred_list)
    print('calibrated mAP test seen:', mAP)
  

    test_predict_unseen = []
    test_label_unseen = []
    for step, (data,label) in enumerate(test_unseen_dataloader):
        with torch.no_grad():
            #data = torch.nn.functional.normalize(data, dim=-1)
            predicts,y = model(data.cuda(), data.cuda())
            test_predict_unseen.append(predicts.cpu())
            test_label_unseen.append(label.cpu())
    test_predict_unseen = torch.cat(test_predict_unseen, dim=0).cpu().numpy()
    test_label_unseen = torch.cat(test_label_unseen, dim=0).cpu().numpy()
    #val_predict = np.argmax(val_predict, axis=-1)
    #print(predicts)
    #acc = sklearn.metrics.top_k_accuracy_score(val_seen_labels, val_predict_seen, k=1)
    pred_list = test_predict_unseen[:, ~(np.all(test_label_unseen == 0, axis=0))]
    label_list = test_label_unseen[:, ~(np.all(test_label_unseen == 0, axis=0))]
    mAP =sklearn.metrics.average_precision_score(label_list,pred_list)
    print('calibrated mAP test unseen:', mAP)
  