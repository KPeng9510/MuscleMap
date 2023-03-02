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
from slowfast.models.head_helper import TransformerBasicHead
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
from sklearn.metrics import confusion_matrix
from einops import rearrange, repeat
from einops.layers.torch import Rearrange






if __name__ == '__main__':
    # ---- data loading


    f = open('/hkfs/work/workspace/scratch/fy2374-musclemap/models/model_set_1/25_09/ablation_module3_2/embeddings/val_seen_embeddings.pkl','rb')
    val_seen_dataset = pkl.load(f)
    f.close()
    f = open('/hkfs/work/workspace/scratch/fy2374-musclemap/models/model_set_1/25_09/ablation_module3_2/embeddings/val_unseen_embeddings.pkl','rb')
    val_unseen_dataset = pkl.load(f)
    f.close()
    f = open('/hkfs/work/workspace/scratch/fy2374-musclemap/models/model_set_1/25_09/ablation_module3_2/embeddings/test_seen_embeddings.pkl','rb')
    test_seen_dataset = pkl.load(f)
    f.close()
    f = open('/hkfs/work/workspace/scratch/fy2374-musclemap/models/model_set_1/25_09/ablation_module3_2/embeddings/test_unseen_embeddings.pkl','rb')
    test_unseen_dataset= pkl.load(f)
    f.close()

    f = open('/hkfs/work/workspace/scratch/fy2374-musclemap/models/model_set_1/25_09/ablation_module3_2/embeddings/val_seen_labels.pkl','rb')
    val_seen_labels = pkl.load(f)
    f.close()
    f = open('/hkfs/work/workspace/scratch/fy2374-musclemap/models/model_set_1/25_09/ablation_module3_2/embeddings/val_unseen_labels.pkl','rb')
    val_unseen_labels = pkl.load(f)
    f.close()
    f = open('/hkfs/work/workspace/scratch/fy2374-musclemap/models/model_set_1/25_09/ablation_module3_2/embeddings/test_seen_labels.pkl','rb')
    test_seen_labels = pkl.load(f)
    f.close()
    f = open('/hkfs/work/workspace/scratch/fy2374-musclemap/models/model_set_1/25_09/ablation_module3_2/embeddings/test_unseen_labels.pkl','rb')
    test_unseen_labels = pkl.load(f)
    f.close()

    val_predict_seen = torch.nn.functional.softmax(torch.cat(val_seen_dataset, dim=0),dim=-1)
    val_predict_seen = val_predict_seen.cpu().numpy()
    val_label_seen = torch.cat(val_seen_labels, dim=0).cpu().numpy()
    print(val_label_seen)
    pred_list = val_predict_seen[:, ~(np.all(val_label_seen == 0, axis=0))]
    label_list = val_label_seen[:, ~(np.all(val_label_seen == 0, axis=0))]
    mAP = [0]
    mAP =sklearn.metrics.average_precision_score(label_list,pred_list, average=None)
    print('calibrated mAP val seen:', np.mean(mAP))

    val_predict_unseen = torch.cat(val_unseen_dataset, dim=0).cpu().numpy()
    val_label_unseen = torch.cat(val_unseen_labels, dim=0).cpu().numpy()
    pred_list = val_predict_unseen[:, ~(np.all(val_label_unseen == 0, axis=0))]
    label_list = val_label_unseen[:, ~(np.all(val_label_unseen == 0, axis=0))]
    mAP =sklearn.metrics.average_precision_score(label_list,pred_list)
    print('calibrated mAP val seen:', mAP)

    test_predict_seen = torch.cat(test_seen_dataset, dim=0).cpu().numpy()
    test_label_seen = torch.cat(test_seen_labels, dim=0).cpu().numpy()
    pred_list = test_predict_seen[:, ~(np.all(test_label_seen == 0, axis=0))]
    label_list = test_label_seen[:, ~(np.all(test_label_seen == 0, axis=0))]
    mAP =sklearn.metrics.average_precision_score(label_list,pred_list)
    print('calibrated mAP val seen:', mAP)

    test_predict_unseen = torch.cat(test_unseen_dataset, dim=0).cpu().numpy()
    test_label_unseen = torch.cat(test_unseen_labels, dim=0).cpu().numpy()
    pred_list = test_predict_unseen[:, ~(np.all(test_label_unseen == 0, axis=0))]
    label_list = test_label_unseen[:, ~(np.all(test_label_unseen == 0, axis=0))]
    mAP =sklearn.metrics.average_precision_score(label_list,pred_list)
    print('calibrated mAP val seen:', mAP)