import pandas as pd

from req import hold_loader, train_loader
from sklearn import preprocessing

lb = preprocessing.LabelBinarizer()
from model import ResNet18mod, ArcFaceLossAdaptiveMargin
from sklearn.preprocessing import normalize

from datas import get_cv, save_model, load_model, customLosses, getsameIndex, get_thetaMat, thetaGraph, vanillaLoder,get_Loss
from torch.utils.data import DataLoader
import os
from config import Config,OptimizationStartegy
from tqdm import tqdm as tqdm
from req import hold2, hold_loader

target_dataset = hold2
target_loader = hold_loader
from torch import autograd
# target_loader=train_loader
# target_dataset=train2
from data.dataset import Dataset
import copy
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
from req import hold
from sklearn.preprocessing import normalize

device = 'cpu'

opt = Config()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if opt.display:
    visualizer = Visualizer()

#     train_dataset = Dataset(opt.train_root, opt.train_list, phase='train', input_shape=opt.input_shape)
#     trainloader = data.DataLoader(train_dataset,
#                                   batch_size=opt.train_batch_size,
#                                   shuffle=True,
#                                   num_workers=opt.num_workers)
# trainloader = target_loader

# identity_list = get_lfw_list(opt.lfw_test_list)
# img_paths = [os.path.join(opt.lfw_root, each) for each in identity_list]

print('{} train iters per epoch:'.format(len(target_dataset)))
if True:
    embeds = []
    maxCat = len(target_dataset['label_group'].unique())
    lb.fit(target_dataset['label_group'])
    target_dataset['label_group'] = target_dataset['label_group'].apply(lambda x: np.array(lb.transform([x])).flatten())
    model = ResNet18mod(32768, maxCat)
    # model = ArcFaceLossAdaptiveMargin(32768, maxCat)
    # save_model(model, opt.checkpoints_path, opt.backbone, 'randomInit')
    with torch.no_grad():
        for img, input_ids, attention_mask, _ in tqdm(target_loader):
            #         img= img
            img = img.to(device)
            feat = model.model(img)  # , input_ids, attention_mask)
            feat = feat.reshape(*feat.shape[:1], -1)
            image_embeddings = feat.detach().cpu().numpy()  # .flatten()
            # image_embeddings = image_embeddings.reshape(*image_embeddings.shape[:1], -1)
            embeds.append(image_embeddings)

    print(image_embeddings.shape)
    image_embeddings = np.concatenate(embeds)
    image_embeddings = image_embeddings.reshape(*image_embeddings.shape[:1], -1)
    image_embeddings = normalize(image_embeddings, axis=1, norm='l2')

    # label=target_dataset[opt.target].reset_index(drop=True)

    vl = vanillaLoder(image_embeddings, target_dataset['label_group'].reset_index(drop=True))
    target_loader = DataLoader(vl, batch_size=512, num_workers=4,shuffle=False)
if opt.loss == 'focal_loss':
    criterion = FocalLoss(gamma=2)
elif opt.loss == 'thetaLoss':
    # same=getsameIndex( 'label_group')
    l = customLosses(margin=15.0, threshold=50.0, same=None)

    criterion = l.calcLoss
#6.b,4.c,3.c,5.probably pic is wrongly taken,2.c
else:
    #criterion = torch.nn.CrossEntropyLoss()
    criterion=torch.nn.BCEWithLogitsLoss(reduction='mean')
    #criterion=torch.nn.CrossEntropyLoss()
if __name__ == '__main__':
    if opt.backbone == 'resnet18':
        model = resnet_face18(use_se=opt.use_se)
    elif opt.backbone == 'resnet34':
        model = resnet34()
    elif opt.backbone == 'resnet50':
        model = resnet50()
    elif opt.backbone == 'arc':
        model = ArcFaceLossAdaptiveMargin(inputDim=32768, outputDim=4096,margins=0.7,s=1)
        if opt.rough:
            initialWeight = copy.deepcopy(model.weight.data)
            initialTheta = torch.tensor(model.theta1.item())
            themata = get_thetaMat(target_loader, model)
            initial_S,initial_D=thetaGraph(themata, target_dataset, 'label_group', opt.miscDirectory + str('randinit3') + "new.png",ret=True)
            model.theta1.data = initialTheta.clone().detach()
            model.weight.data = initialWeight.clone().detach()
            initial_L=get_Loss(target_loader, model, criterion)
            #save_model(model, opt.checkpoints_path, opt.backbone, 'randomInit')
            print("test start initial loss:{}".format(initial_L))
            # for i in range(2):
            #
            #     model.theta1.data = torch.tensor(initialTheta)
            #     model.weight.data = initialWeight
            #     initial_L = get_Loss(target_loader, model, criterion)
            #     print("updated loss:{}".format(initial_L))




    else:
        model = ResNet18mod(32768, 8811)  # len(target_dataset.iloc[0].label_group))
        # save_model(model, opt.checkpoints_path, opt.backbone, 'randomInit')
    if opt.preTraining:  # trained weights available
        WGT = opt.load_model_path
        model = load_model(model, WGT)
        for t in [0.2+i*0.05 for i in range(14)]:
              get_cv(target_loader, model, target_dataset,threshold=t)
        #get_cv(target_loader, model, target_dataset,threshold=0.7)
        themata = get_thetaMat(target_loader, model)
        thetaGraph(themata, target_dataset, 'label_group',
                          opt.miscDirectory + str('init') + "new3f2.png")

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
        optimizer = torch.optim.SGD(opt.customRate(model),
                                     weight_decay=opt.weight_decay)
        optStrategy=OptimizationStartegy(model,[opt.lr1,opt.lr2])
        lr1, lr2=opt.lr1,opt.lr2
    else:
        optimizer = torch.optim.Adam([{'params': model.parameters()}, {'params': metric_fc.parameters()}],
                                     lr=opt.lr, weight_decay=opt.weight_decay)
    scheduler = StepLR(optimizer, step_size=opt.lr_step, gamma=0.1)

    start = time.time()
    iters = 0

    #forlr
    if opt.rough:
        lastLoss=0
        index1,index2,index3=-1,-1,0
        steps = 10
        lrMain = [0.001 * 10**i for i in range(steps)]
        lrList=np.array([[0, 0, initial_L, initialTheta, initial_S, initial_D]])
        temp = pd.DataFrame(
            {'index1': lrList[:, 0], 'index2': lrList[:, 1], 'loss': lrList[:, 2], 'theta': lrList[:, 3],
             's': lrList[:, 4], 'd': lrList[:, 5]})
        temp = temp.astype('float')
        temp.to_csv(opt.miscDirectory + 'learning.csv', mode='w', header=True)

    i = 0
    decreaseLoss=True
    while i <opt.max_epoch:

        if i==0:
            lastLoss=1
            w1, w2, w3, w4 = 1, 100, 1, 100
        else :
            if i==1:optStrategy.lastLoss=loss.item()
            if (i-1)%5==0 and i >5:
                # model.theta1.data=saveTheta
                # model.weight.data=saveWeight
                loss1=get_Loss(target_loader, model, criterion)
                print('loss={}'.format(float(loss1)))
                lrChange=optStrategy.mainStrategy(loss1,model.theta1.data.clone().detach(),s,d)
                #print('inside lr')
                if lrChange is not None:
                    lr1, lr2=tuple(lrChange)
                    # optimizer = torch.optim.SGD([{'params': model.theta1, 'lr': lr1}, {'params': mode   l.weight, 'lr': lr2}],
                    #                         weight_decay=opt.weight_decay)
                if optStrategy.msg.find('|old')!=-1:

                    if decreaseLoss:
                        tmp=model.theta1.data.clone().detach()
                        model.theta1.data=optStrategy.last[0].clone().detach()
                        themata = get_thetaMat(target_loader, model)
                        # s, d = thetaGraph(themata, target_dataset, 'label_group',
                        #                   opt.miscDirectory + str(i) + "new3f2.png", ret=True)
                        thetaGraph(themata, target_dataset, 'label_group',
                                   opt.miscDirectory + str(i) + "new3f2.png",title='graph with loss '+str(optStrategy.lastLoss))
                        model.theta1.data =tmp.clone().detach()

                        save_model(model, opt.checkpoints_path, opt.backbone + 'retrained', i)
                        decreaseLoss=False

                else:decreaseLoss=True

                print((model.theta1.data%360).clone().detach())


        if opt.rough and (i==0 or (i-1)%opt.save_interval==0):#learnign rate evaluator

            if i>opt.save_interval:lrList=np.append(lrList, [[lrMain[index1], lrMain[index2], closs - initial_L, model.theta1.item() - initialTheta, s - initial_S, d - initial_D]], axis=0)
            if i!=1:
                index2 += 1
                if index2%steps==0:
                    index1+=1
                    index2 = 0
                    if i>opt.save_interval:
                        temp=pd.DataFrame({'index1':lrList[1:,0],'index2':lrList[1:,1],'loss':lrList[1:,2],'theta':lrList[1:,3],'s':lrList[1:,4],'d':lrList[1:,5]})
                        temp=temp.astype('float')
                        temp.to_csv(opt.miscDirectory+'learning.csv',mode='a',header=False)
                        lrList = np.array([[0, 0, initial_L, initialTheta, initial_S, initial_D]])
                        model.theta1.data = initialTheta.clone().detach()
                        model.weight.data = initialWeight.clone().detach()
                        print(index1,index2,torch.tensor(model.theta1.data),torch.mean(model.weight.data),get_Loss(target_loader, model, criterion))
                        print('lr File updated')
                optimizer = torch.optim.SGD(opt.customRate(model,lrMain[index1],lrMain[index2]),
                                        weight_decay=opt.weight_decay)
                model.theta1.data=initialTheta.clone().detach()
                model.weight.data=initialWeight.clone().detach()


                #print(index1,index2,torch.tensor(model.theta1.data),torch.mean(model.weight.data),get_Loss(target_loader, model, criterion))
                i=0


        scheduler.step()

        model.train()
        for image, label in tqdm(target_loader):
                data_input, label = image, label
                data_input = data_input.to(device)
                output = model(data_input)
                lb=torch.matmul(label,label.T)
                # lb=torch.ones(lb.shape[0],lb.shape[0])
                # lb = torch.zeros(lb.shape[0], lb.shape[0])
                #lb = lb - torch.eye(lb.shape[0])
                crossLoses=model.marginLoss(output,lb.T,w1=w1, w2=w2 , w3=w3, w4=w4)
                #crossLoses=model.arcFaceappl(output,lb.T)
               # crossLoses=model.crosscLoss(output,lb.T)
                #print(str(torch.isnan(output).any()), str(torch.isnan(crossLoses).any()))


                #label = torch.argmax(label, dim=1)
                # idx,idy= getsameIndex(label)
                # label=torch.zeros(len(label),len(label))
                # label[[idx],[idy]]=1

                #print(get_Loss(target_loader, model, criterion))
                loss = criterion(crossLoses, lb.type_as(crossLoses))
                print('epoch:{},loss:{}:'.format(i, loss.item()))

                #print(loss.item(),get_Loss(target_loader, model, criterion))
                optimizer.zero_grad()

                loss.backward()
                model.theta1.data = model.theta1.data.clamp(0, 90)
                model.theta1.register_hook(lambda grad: torch.clamp(grad, -50, 50))
                optimizer.step()

                #print(torch.max(model.weight.grad),torch.min(model.weight.grad),model.theta1.grad)
                #print(torch.max(model.weight), torch.min(model.weight))
                # print(model.weight)
                #print(torch.mean(model.weight),model.theta1*57.2958)
                iters += 1




        if iters % opt.print_freq ==479:
                    output = output.data.cpu().numpy()
                    output = np.argmax(output, axis=1)
                    # label = torch.argmax(label, dim=1)
                    label = label.data.cpu().numpy()
                    # print(output)
                    # print(label)
                    acc = np.mean((output == label).astype(int))
                    speed = opt.print_freq / (time.time() - start)
                    time_str = time.asctime(time.localtime(time.time()))
                    # print('{} train epoch {}  {} iters/s loss {} acc {}'.format(time_str, i, speed, loss.item(),
                    #                                                             acc))
        if opt.display :
                visualizer.display_current_results(iters, loss.item(), name='loss')
                visualizer.display_current_results(iters, model.theta1.item(), name='theta')
                visualizer.display_current_results(iters, lr1, name='thetaLR')
                visualizer.display_current_results(iters, lr2, name='weightsLR')
                visualizer.display_current_results(iters, optStrategy.lastLoss, name='bestLoss')


        if i % opt.save_interval == 0 and i != 0:
                try:
                        themata = get_thetaMat(target_loader, model)
                        s, d = thetaGraph(themata, target_dataset, 'label_group',
                                          opt.miscDirectory + str(i) + "new3f2.png", ret=True)
                        # thetaGraph(themata, target_dataset, 'label_group',
                        #            opt.miscDirectory + str(i) + "new3f2.png")
                        visualizer.display_current_results(iters, s, name='sameMean')
                        visualizer.display_current_results(iters, d, name='diffMean')
                        #save_model(model, opt.checkpoints_path, opt.backbone + 'retrained', i)
                        present = time.time()
                        print("{} minute passed".format((present - start) / 60))
                        closs=loss.item()
                except:
                    s,d,closs=-999.0,-999.0,-999.0

        model.eval()

        i+=1
        # acc = lfw_test(model, img_paths, identity_list, opt.lfw_test_list, opt.test_batch_size)


