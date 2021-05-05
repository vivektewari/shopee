from req import hold_loader,train_loader
from req2 import ResNet18mod
from req import hold2,train2
from datas import get_cv,save_model
import os
from config import Config
from tqdm import tqdm as tqdm
target_loader=hold_loader
target_dataset=hold2
target_loader=train_loader
target_dataset=train2
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

device = 'cpu'





if __name__ == '__main__':

    opt = Config()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if opt.display:
        visualizer = Visualizer()

    #     train_dataset = Dataset(opt.train_root, opt.train_list, phase='train', input_shape=opt.input_shape)
    #     trainloader = data.DataLoader(train_dataset,
    #                                   batch_size=opt.train_batch_size,
    #                                   shuffle=True,
    #                                   num_workers=opt.num_workers)
    trainloader = target_loader

    # identity_list = get_lfw_list(opt.lfw_test_list)
    # img_paths = [os.path.join(opt.lfw_root, each) for each in identity_list]

    print('{} train iters per epoch:'.format(len(trainloader)))

    if opt.loss == 'focal_loss':
        criterion = FocalLoss(gamma=2)
    else:
        criterion = torch.nn.CrossEntropyLoss()

    if opt.backbone == 'resnet18':
        model = resnet_face18(use_se=opt.use_se)
    elif opt.backbone == 'resnet34':
        model = resnet34()
    elif opt.backbone == 'resnet50':
        model = resnet50()
    else:
        model = ResNet18mod(32768, len(target_dataset.iloc[0].label_group))

    if opt.metric == 'add_margin':
        metric_fc = AddMarginProduct(512, opt.num_classes, s=30, m=0.35)
    elif opt.metric == 'arc_margin':
        metric_fc = ArcMarginProduct(512, opt.num_classes, s=30, m=0.5, easy_margin=opt.easy_margin)
    elif opt.metric == 'sphere':
        metric_fc = SphereProduct(512, opt.num_classes, m=4)
    else:
        metric_fc = None  # nn.Linear(512, opt.num_classes)

    # view_model(model, opt.input_shape)
    # print(model)
    model.to(device)
    # model = DataParallel(model)
    #     metric_fc.to(device)
    #     metric_fc = DataParallel(metric_fc)

    if opt.optimizer == 'sgd':
        #         optimizer = torch.optim.SGD([{'params': model.parameters()}, {'params': metric_fc.parameters()}],
        #                                     lr=opt.lr, weight_decay=opt.weight_decay)
        optimizer = torch.optim.SGD(model.parameters(),
                                    lr=opt.lr, weight_decay=opt.weight_decay)
    else:
        optimizer = torch.optim.Adam([{'params': model.parameters()}, {'params': metric_fc.parameters()}],
                                     lr=opt.lr, weight_decay=opt.weight_decay)
    scheduler = StepLR(optimizer, step_size=opt.lr_step, gamma=0.1)

    start = time.time()
    iters = 0

    for i in range(opt.max_epoch):
        scheduler.step()

        model.train()
        for image, input_ids, attention_mask, label in tqdm(trainloader):
            data_input, label = image, label
            data_input = data_input.to(device)
            label = label.to(device).long()
            output = model(data_input, label)
            # output = metric_fc(feature, label)
            label=torch.argmax(label, dim=1)
            loss = criterion(output, label)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            iters += 1

            if iters % opt.print_freq == 0:
                output = output.data.cpu().numpy()
                output = np.argmax(output, axis=1)
                label = label.data.cpu().numpy()
                # print(output)
                # print(label)
                acc = np.mean((output == label).astype(int))
                speed = opt.print_freq / (time.time() - start)
                time_str = time.asctime(time.localtime(time.time()))
                print('{} train epoch {}  {} iters/s loss {} acc {}'.format(time_str, i, speed, loss.item(),
                                                                                   acc))
                if opt.display:
                    visualizer.display_current_results(iters, loss.item(), name='train_loss')
                    visualizer.display_current_results(iters, acc, name='train_acc')

                start = time.time()

        if i % opt.save_interval == 0 or i == opt.max_epoch:

            save_model(model, opt.checkpoints_path, opt.backbone, i)
            get_cv(trainloader, model, target_dataset)

        model.eval()
        #acc = lfw_test(model, img_paths, identity_list, opt.lfw_test_list, opt.test_batch_size)
        if opt.display:
            visualizer.display_current_results(iters, acc, name='test_acc')

