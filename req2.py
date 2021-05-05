import torch.nn.functional as F
import torch.nn as nn
import torchvision
import torch
import math
import numpy as np


class ArcMarginProduct(nn.Module):
    r"""Implement of large margin arc distance: :
        Args:
            in_features: size of each input sample
            out_features: size of each output sample
            s: norm of input feature
            m: margin
            cos(theta + m)
        """

    def __init__(self, in_features, out_features, s=30.0, m=0.50, easy_margin=False):
        super(ArcMarginProduct, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.s = s
        self.m = m
        self.weight = Parameter(torch.FloatTensor(out_features, in_features))
        nn.init.xavier_uniform_(self.weight)

        self.easy_margin = easy_margin
        self.cos_m = math.cos(m)
        self.sin_m = math.sin(m)
        self.th = math.cos(math.pi - m)
        self.mm = math.sin(math.pi - m) * m

    def forward(self, input, label):
        # --------------------------- cos(theta) & phi(theta) ---------------------------
        cosine = F.linear(F.normalize(input), F.normalize(self.weight))
        sine = torch.sqrt((1.0 - torch.pow(cosine, 2)).clamp(0, 1))
        phi = cosine * self.cos_m - sine * self.sin_m
        if self.easy_margin:
            phi = torch.where(cosine > 0, phi, cosine)
        else:
            phi = torch.where(cosine > self.th, phi, cosine - self.mm)
        # --------------------------- convert label to one-hot ---------------------------
        # one_hot = torch.zeros(cosine.size(), requires_grad=True, device='cuda')
        one_hot = torch.zeros(cosine.size(), device='cuda')
        one_hot.scatter_(1, label.view(-1, 1).long(), 1)
        # -------------torch.where(out_i = {x_i if condition_i else y_i) -------------
        output = (one_hot * phi) + (
                    (1.0 - one_hot) * cosine)  # you can use torch.where if your torch.__version__ is 0.4
        output *= self.s
        # print(output)

        return output


class ArcFaceLossAdaptiveMargin(nn.modules.Module):
    def __init__(self, inputDim, outputDim, margins=0.5, s=30.0):
        super().__init__()
        # self.crit = DenseCrossEntropy()
        self.weight = nn.Parameter(torch.FloatTensor(outputDim, inputDim))
        self.reset_parameters()
        self.s = s
        self.margins = margins
        self.lr1 = nn.Linear(inputDim, outputDim)
        self.outputDim = outputDim

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)

    def forward(self, input1, labels):
        x = self.lr1(input1)
        # logits = logits.float()
        cosine = F.linear(F.normalize(input1), F.normalize(self.weight))
        # ms = []
        # ms = self.margins[labels1.cpu().numpy()]
        ms = self.margins
        cos_m = np.cos(ms)
        sin_m = np.sin(ms)
        th = np.cos(math.pi - ms)
        mm = np.sin(math.pi - ms) * ms

        sine = torch.sqrt(1.0 - torch.pow(cosine, 2))
        phi = cosine * cos_m - sine * sin_m
        phi = torch.where(cosine > th, phi, cosine - mm)
        output = (labels * phi) + ((1.0 - labels) * cosine)
        output *= self.s
        # loss = self.crit(output, labels)
        # return loss
        return output
    def predict(self, input1):
        x = self.lr1(input1)
        # logits = logits.float()
        cosine = F.linear(F.normalize(input1), F.normalize(self.weight))
        return cosine



class ResNet18mod(nn.Module):
    def __init__(self, inputDim, outputDim):
        super(ResNet18mod, self).__init__()
        model_ft = torchvision.models.resnet18(pretrained=True)
        for param in model_ft.parameters(): param.requires_grad = False
        self.model = nn.Sequential(*list(nn.Sequential(*list(model_ft.children())[:-1]))[:-1])

        self.arc = ArcFaceLossAdaptiveMargin(inputDim, outputDim)

    def forward(self, image, labels1):
        image_embeddings = self.model(image)
        image_embeddings = image_embeddings.reshape(*image_embeddings.shape[:1], -1)
        output = self.arc(image_embeddings, labels1)
        return output
    def predict(self, image):
        image_embeddings = self.model(image)
        image_embeddings = image_embeddings.reshape(*image_embeddings.shape[:1], -1)
        output = self.arc.predict(image_embeddings)
        return output
