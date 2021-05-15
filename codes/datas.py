#!pip install -v --disable-pip-version-check --no-cache-dir --global-option="--cpp_ext" --global-option="--cuda_ext"
from sklearn.preprocessing import normalize
import matplotlib.pyplot as plt
from itertools import permutations,combinations
import pandas as pd
import seaborn as sns
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

import datas

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

def getMetric2(o, v):
        n = len(np.intersect1d(v, o))
        return 2 * n / (len(v) + len(o))
class vanillaLoder(Dataset):
    def __init__(self, image,label):
        self.label = label
        self.image=image
    def __getitem__(self, index):
        return torch.tensor(self.image[index]),  torch.tensor(self.label[index])
    def __len__(self):
        return self.label.shape[0]



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

def holdOut(train,groupLabel,splits):
    df_train = train

    gkf = GroupKFold(n_splits=splits)

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

class customLosses():
    def __init__(self,margin=20.0,threshold=50.0,same=None,device='cpu'):
        self.margin=margin
        self.th=threshold
        self.same=same
        self.device=device
    def thetaLoss(self,thetas):
        same=self.same
        
        factorDiv=(thetas.shape[0]*thetas.shape[0]-thetas.shape[0])/2
        sameMatrix=thetas[same[0],same[1]]
        zerotensor=torch.tensor(0.0)
        sameLoss=torch.where(sameMatrix>(self.th-self.margin),(sameMatrix-(self.th+self.margin)),zerotensor)
        diffLoss=torch.where(thetas<(self.th+self.margin),self.th+self.margin-thetas,zerotensor)
        diffLoss = torch.triu(diffLoss, diagonal=1)
        diffLoss[same[0], same[1]] = 0
        lossSame=torch.sum(torch.square(sameLoss))
        lossDiff = torch.sum(torch.square(diffLoss))
        loss=(lossSame+lossDiff)/factorDiv
        loss=torch.sum(torch.square(thetas))
        return loss
    def calcTheta(self,image_embeddings):
        #image_embeddings = np.array(image_embeddings)

        image_embeddings = F.normalize(image_embeddings, dim=1)

        image_embeddings = image_embeddings.to(self.device)
        cts = torch.matmul(image_embeddings, image_embeddings.T).T
        cts=torch.clip(cts,-0.95,0.95)
        cts=torch.acos(cts) * 57.29
        return cts
    def calcLoss(self,image_embeddings):
        theta=self.calcTheta(image_embeddings)
        thetaLoss=self.thetaLoss(theta)
        return thetaLoss
            


def getsameIndex(label):
    #label = torch.argmax(label, dim=1)
    #a0 = data[:]
    a0=pd.DataFrame(data={'index':[i for i in range(len(label))],'col':label})
   # a0[labelVar] = a0[labelVar].apply(lambda x: np.argmax(x))
    #a0['index'] = [i for i in range(a0.shape[0])]
    a = a0.groupby('col')['index'].apply(list).reset_index()

    #a1 = pd.DataFrame({'index': a},index=a)
    dict1 = a.to_dict()['index']

    #maxRows = data.shape[0]
    # all = [(i, j) for i, j in list(combinations([i for i in range(maxRows)], 2))]
    same = []
    # sameP=[]
    #print("distribution among same and different group starting")
    for key in dict1.keys(): same.extend([i for i in combinations(dict1[key], 2)])
    # for key in dict1.keys(): sameP.extend([i for i in permutations(dict1[key], 2)])
    a,b = zip(*same)
    # diff = list(set(all) - set(sameP) )#- set([(i, i) for i in range(maxRows)])
    # c,d = zip(*diff)
    return (a,b)#,(c,d)

def get_thetaMat(data_loader, model,modelOutputOnly=False):
    embeds = []
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    with torch.no_grad():
        for img,label in tqdm(data_loader):
            #         img= img
            img = img.to(device)
            feat = model.predict(img)  # , input_ids, attention_mask)
            image_embeddings = feat.detach().cpu().numpy()  # .flatten()
            # image_embeddings = image_embeddings.reshape(*image_embeddings.shape[:1], -1)
            embeds.append(image_embeddings)
    image_embeddings = np.concatenate(embeds)
    if modelOutputOnly:return torch.acos(torch.tensor(image_embeddings)) * 57.29
    #print('Nan count is:'+str(np.isnan(image_embeddings).any()))
    del embeds
    #print(image_embeddings.shape)
    _ = gc.collect()
    #print('image embeddings shape', image_embeddings.shape)
    image_embeddings = np.array(image_embeddings)

    image_embeddings = normalize(image_embeddings, axis=1)

    image_embeddings = torch.from_numpy(image_embeddings).to(device)

    CHUNK = 512

    CTS = len(image_embeddings) // CHUNK
    if len(image_embeddings) % CHUNK != 0: CTS += 1
    finalMatrix=torch.zeros( len(image_embeddings),  len(image_embeddings))
    for j in range(CTS):

        a = j * CHUNK
        b = (j + 1) * CHUNK
        b = min(b, len(image_embeddings))
        #print('chunk', a, 'to', b)

        cts = torch.matmul(image_embeddings, image_embeddings[a:b].T).T

        finalMatrix[[i for i in range(a,b)],:]=cts
    finalMatrix=torch.clamp(finalMatrix,min=-1,max=1)
    finalMatrix=torch.acos(finalMatrix) * 57.29
    #print("theta matrix prepared")
    #print('min,max,average theta value are:'+str(torch.min(finalMatrix))+","+str(torch.max(finalMatrix))+","+str(torch.mean(finalMatrix)))
    return  finalMatrix
def thetaGraph(thetaMatrix, data, labelVar, saveLoc,outputCross=True,ret=False):
    maxCol = thetaMatrix.shape[1]
    s = data[labelVar].apply(lambda x: np.argmax(x))
    if outputCross:
        idx_i, idx_j=getsameIndex(s)
        all = [(i, j) for i, j in list(combinations([i for i in range(maxCol)], 2))]
    else:
        idx_i, idx_j = [i for i in range(thetaMatrix.shape[0])], list(s)
        all = [(i, j) for i in range(thetaMatrix.shape[0]) for j in range(thetaMatrix.shape[1])]

        #same=[]
    #print("distribution among same and different group starting")
    same = list(zip(idx_i, idx_j))
    sameList = thetaMatrix[idx_i, idx_j]
    diff = list(set(all) - set(same))
    idx_i, idx_j = zip(*diff)
    diffList= thetaMatrix[idx_i, idx_j]
    #print("avg_val for same and diff is"+str(torch.mean(sameList))+','+str(torch.mean(diffList)))

    if ret:return torch.mean(sameList),torch.mean(diffList)
    else:
        p=plotHist([diffList,sameList],bins= 100,names=['different','same'],title='theta distribution of same and different group' )
        p.savefig(saveLoc)

def get_cv(data_loader, model, data,threshold=0.7):
    embeds = []
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    with torch.no_grad():
        for img, _ in tqdm(data_loader):
            #         img= img
            img = img.to(device)
            feat = model.predict(img)  # , input_ids, attention_mask)
            image_embeddings = feat.detach().cpu().numpy()  # .flatten()
            # image_embeddings = image_embeddings.reshape(*image_embeddings.shape[:1], -1)
            embeds.append(image_embeddings)
    image_embeddings = np.concatenate(embeds)
    del embeds
    print(image_embeddings.shape)
    _ = gc.collect()
    print('image embeddings shape', image_embeddings.shape)
    image_embeddings = np.array(image_embeddings)

    image_embeddings = normalize(image_embeddings, axis=1, norm='l2')

    preds = []
    CHUNK = 512
    data['preds2'] = 0
    predInd = data.columns.get_loc('preds2')
    tarInd = data.columns.get_loc('target')
    print('Finding similar images...')
    CTS = len(image_embeddings) // CHUNK
    image_embeddings = torch.from_numpy(image_embeddings).to(device)
    if len(image_embeddings) % CHUNK != 0: CTS += 1

    for j in range(CTS):

        a = j * CHUNK
        b = (j + 1) * CHUNK
        b = min(b, len(image_embeddings))
        print('chunk', a, 'to', b)

        cts = torch.matmul(image_embeddings, image_embeddings[a:b].T).T
        storage = []
        for k in range(b - a):

            #         print(sorted(cts[k,], reverse=True))
            IDX = torch.where(cts[k,] > threshold)[0]
            if torch.cuda.is_available():o = IDX.to('cpu')
            else:o = IDX
            v = data.iloc[o].posting_id.values
            storage.append(v)

            del o
        data.iloc[a :b, predInd] = [getMetric2(x, y) for x,y in list(zip(storage,data.iloc[a : b, tarInd])) ]
        del v
    # data['preds2'] = np.array([data.iloc[IDX].posting_id.values for IDX in preds])
    if 'target' in data.columns: print("m={},0CV for image {} :".format(threshold,round(data['preds2'].mean(), 3)))
    return image_embeddings, data
def get_cv2(data_loader, model, data):
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
def load_model(model, model_file):
    if torch.cuda.is_available()  :state_dict = torch.load(model_file)
    else:state_dict = torch.load(model_file, map_location=torch.device('cpu'))
    if "model_state_dict" in state_dict.keys():
        state_dict = state_dict["model_state_dict"]
    state_dict = {k[7:] if k.startswith('module.') else k: state_dict[k] for k in state_dict.keys()}
    model.load_state_dict(state_dict, strict=True)
    #else : model.load_state_dict( torch.load(state_dict ,  map_location=torch.device('cpu')))
    print(f"loaded {model_file}")
    model.eval()
    return model
def plotHist(x,bins,names,title):
    plt.clf()
    colors = ['#56B4E9','#E69F00',  '#F0E442', '#009E73', '#D55E00']
    plt.hist(x, bins=bins,
             color=colors[0:len(x)], label=names,density=True)
    # Add labels
    plt.legend(prop ={'size': 10})
    plt.title(title)
    plt.xlabel('counts')
    plt.ylabel('values')
    plt.grid()
    plt.xticks(np.arange(0, 180, 10))

    return plt


if __name__ == 'datas':

    from unittest import TestCase
    class TestcustomLosses(TestCase):
        a=customLosses()
        testmat=torch.tensor([[0.2,0.8,0.4,0.1],[0.97,0.4,0.4,0.1]])
        same=[0,1,0],[2,1,0]
        diff=[0,0,1,1,1],[1,3,0,2,3]

        def test_thetaLoss(self):
            print('starting')
            #l=self.a.thetaLoss(self.testmat,self.same,self.diff)


    class TestMisc(TestCase):

        def test_getsamediffIndex(self):
            from req import train2
            a = customLosses()
            #l = self.a.thetaLoss(self.testmat, self.same, self.diff)
            label=train2.label_group
            t1=getsameIndex( label)

            c=0




