#!pip install -v --disable-pip-version-check --no-cache-dir --global-option="--cpp_ext" --global-option="--cuda_ext"
from sklearn.preprocessing import normalize
import os
import cv2
#from apex.parallel import DistributedDataParallel
import numpy as np
import pandas as pd
import albumentations
import torchvision
import torch
from torch.utils.data import Dataset
from sklearn.model_selection import GroupKFold
from sklearn.model_selection import train_test_split, StratifiedKFold
import os

os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
from transformers import *
import gc
import cv2
import math
import time
import pickle
import random
import argparse
# import albumentations
import numpy as np, gc
import pandas as pd
from tqdm import tqdm as tqdm
from sklearn.metrics import cohen_kappa_score, confusion_matrix
#from commonFuncs import packing
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader, Dataset
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.backends import cudnn
from torchvision import  models
import copy
import gc,ast
import math
from tqdm import tqdm
# import apex
# from apex import amp
# from apex.parallel import DistributedDataParallel

# #from dataset import LandmarkDataset, get_df, get_transforms
# from util import global_average_precision_score, GradualWarmupSchedulerV2
# from models import DenseCrossEntropy, Swish_module
# from models import ArcFaceLossAdaptiveMargin, Effnet_Landmark, RexNet20_Landmark, ResNest101_Landmark, load_model


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--kernel-type', type=str, required=False)  # True
    parser.add_argument('--data-dir', type=str,
                        default='/home/pooja/Downloads/shopee-product-matching/')
    parser.add_argument('--train-step', type=int, default=0)  # True,required=True,
    parser.add_argument('--image-size', type=int, required=False)  # True
    parser.add_argument("--local_rank", type=int)
    parser.add_argument('--enet-type', type=str, required=False)  # True
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--num-workers', type=int, default=32)
    parser.add_argument('--init-lr', type=float, default=1e-4)
    parser.add_argument('--n-epochs', type=int, default=15)
    parser.add_argument('--start-from-epoch', type=int, default=1)
    parser.add_argument('--stop-at-epoch', type=int, default=999)
    parser.add_argument('--use-amp', action='store_false')
    parser.add_argument('--DEBUG', action='store_true')
    parser.add_argument('--model-dir', type=str, default='./weights')
    parser.add_argument('--log-dir', type=str, default='./logs')
    parser.add_argument('--CUDA_VISIBLE_DEVICES', type=str, default='-1')  # '0,1,2,3,4,5,6,7'
    parser.add_argument('--fold', type=int, default=0)
    parser.add_argument('--load-from', type=str, default='')
    args, _ = parser.parse_known_args()
    return args


def set_seed(seed=0):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
def getMetric(col):
    def f1score(row):
        n = len( np.intersect1d(row.target,row[col]) )
        return 2*n / (len(row.target)+len(row[col]))
    return f1score
class LandmarkDataset(Dataset):
    def __init__(self, csv, split, mode, transform=None,tokenizer =None):

        self.csv = csv.reset_index()
        self.split = split
        self.mode = mode
        self.transform = transform
        self.tokenizer = tokenizer #vivAdd

    def __len__(self):
        return self.csv.shape[0]
    def trialImage(self, index):
        import matplotlib.pyplot as plt
        row = self.csv.iloc[index]
        text = row.title
        image = cv2.imread(row.filepath)
        #print(len(image))
        res0 = self.transform(image=image)
        image0 = res0['image'].astype(np.float32)
        image = image0#.transpose(2, 0, 1)
        text = self.tokenizer(text, padding='max_length', truncation=True, max_length=16, return_tensors="pt")
        input_ids = text['input_ids'][0]
        attention_mask = text['attention_mask'][0]
        plt.imshow(image)
        plt.savefig("mygraph.png")
        image = image[:, :, ::-1]
    def __getitem__(self, index):
        row = self.csv.iloc[index]

        text = row.title
        #print(text)
        image = cv2.imread(row.filepath)
        image = image[:, :, ::-1]

        res0 = self.transform(image=image)
        image0 = res0['image'].astype(np.float32)
        image = image0.transpose(2, 0, 1)

        #text = self.tokenizer(text, padding='max_length', truncation=True, max_length=16, return_tensors="pt")
        input_ids = 0#text['input_ids'][0]
        attention_mask =0# text['attention_mask'][0]

        if self.mode == 'test':
            return torch.tensor(image), input_ids, attention_mask,_
        else:
            return torch.tensor(image), input_ids, attention_mask, torch.tensor(row.label_group,dtype=torch.long)

def get_transforms(image_size=256):

    return  albumentations.Compose([
                albumentations.Resize(image_size, image_size),
                albumentations.Normalize()
            ])

def holdOut(train,groupLabel):
    df_train = train

    gkf = GroupKFold(n_splits=5)

    for train_index, test_index in gkf.split(df_train,y=None, groups=df_train[groupLabel]):
        X_train, X_test = df_train.iloc[train_index], df_train.iloc[test_index]

    return  X_train, X_test

def get_df(kernel_type, data_dir, train_step,nrows=None):


    # df = pd.read_csv(data_dir+'/train.csv',nrows=nrows)
    # else:df = pd.read_csv(data_dir+'/train.csv')

    if train_step == 0:
        if nrows is not None:df_train = pd.read_csv(os.path.join(data_dir, 'train.csv'),nrows=nrows)#.drop(columns=['url'])
        else:df_train = pd.read_csv(os.path.join(data_dir, 'train.csv'))



    tmp = df_train.groupby('label_group').posting_id.agg('unique').to_dict()
    df_train['target'] = df_train.label_group.map(tmp)

    out_dim =0# df.landmark_id.nunique()

    return df_train, out_dim
def get_cv(data_loader, model, data):
    embeds = []
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    with torch.no_grad():
        for img, input_ids, attention_mask, _ in tqdm(data_loader):
            #         img= img
            img = img.to(device)
            feat = model.predict(img)  # , input_ids, attention_mask)
            image_embeddings = feat.detach().cpu().numpy()  # .flatten()
            #image_embeddings = image_embeddings.reshape(*image_embeddings.shape[:1], -1)
            embeds.append(image_embeddings)

    print(image_embeddings.shape)
    image_embeddings = np.concatenate(embeds)
    del embeds
    _ = gc.collect()
    print('image embeddings shape', image_embeddings.shape)
    image_embeddings = np.array(image_embeddings)

    image_embeddings = normalize(image_embeddings, axis=1, norm='l2')

    preds = []
    CHUNK = 1024

    print('Finding similar images...')
    CTS = len(image_embeddings) // CHUNK
    if len(image_embeddings) % CHUNK != 0: CTS += 1
    for j in range(CTS):

        a = j * CHUNK
        b = (j + 1) * CHUNK
        b = min(b, len(image_embeddings))
        print('chunk', a, 'to', b)

        cts = np.matmul(image_embeddings, image_embeddings[a:b].T).T

        for k in range(b - a):
            #         print(sorted(cts[k,], reverse=True))
            IDX = np.where(cts[k,] > 0.6)[0]
            o = data.iloc[IDX].posting_id.values
            preds.append(o)
    data['preds2'] = preds
    if 'target' in data.columns: print("CV for image :", round(data.apply(getMetric('preds2'), axis=1).mean(), 3))
    return image_embeddings, data
def save_model(model, save_path, name, iter_cnt):
    save_name = os.path.join(save_path, name + '_' + str(iter_cnt) + '.pth')
    torch.save(model.state_dict(), save_name)
    return save_name