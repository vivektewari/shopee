from req import hold_loader,train_loader
from model import ResNet18mod,ArcFaceLossAdaptiveMargin
from req import hold2,train2
from datas import get_cv,save_model,load_model,get_thetaMat,thetaGraph
import os
from config import Config
from tqdm import tqdm as tqdm
target_loader=hold_loader
target_dataset=hold2
from req3 import target_dataset,target_loader,maxCat
# target_loader=train_loader
# target_dataset=train2
from data.dataset import Dataset

import torch
from torch.utils import data
import torch.nn.functional as F
from models import *
import torchvision
from utils.visualizer import Visualizer
from utils.view_model import view_model
import torch
import numpy as np
import random
import time
# from config.config import Config
from torch.nn import DataParallel
from torch.optim.lr_scheduler import StepLR


# from test import *
import gc
import os

from data.dataset import Dataset

import torch
from torch.utils import data
import torch.nn.functional as F
from models import *
import torchvision
from utils.visualizer import Visualizer
from utils.view_model import view_model
import torch
import numpy as np
import random
import time
# from config.config import Config
from torch.nn import DataParallel
from torch.optim.lr_scheduler import StepLR


# from test import *

import torch.nn.functional as F

from sklearn.preprocessing import normalize







if __name__ == '__main__':

    opt = Config()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")








    criterion = torch.nn.CrossEntropyLoss()


    model = ResNet18mod(32768,8811)# len(target_dataset.iloc[0].label_group))
    model=ArcFaceLossAdaptiveMargin(32768, maxCat )
    for t in [0.6, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95]:
        for WGT in ['arc_15']:#['arc_randomInit','arc_0','arc_5','arc_10','arc_15']:# ['0_randomInit.pth','0_0.pth','0_10.pth','0_20.pth','0_30.pth']:#
            #WGT = opt.load_model_path
            model = load_model(model, '/home/pooja/PycharmProjects/pythonProject1/trainedModels/'+WGT+".pth")
            print(WGT)
            get_cv(target_loader, model, target_dataset,t)
            #themata=get_thetaMat(target_loader, model)#,True
            #thetaGraph(themata,target_dataset,'label_group',opt.miscDirectory+WGT+"_crossProduct.png")#,False