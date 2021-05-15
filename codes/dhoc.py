from random import seed,random
import numpy as np
import math,torch
import matplotlib.pyplot as plt
from datas import plotHist

cosine=torch.tensor([random()*np.power(-1,i) for i in range(1000)])
ms = 0
angle=30
ms=np.cos(angle*0.0174533)
cos_m = np.cos(ms)
sin_m = np.sin(ms)
th = np.cos(math.pi - ms)
mm = np.sin(math.pi - ms) * ms

sine = torch.sqrt(1.0 - torch.pow(cosine, 2))
phi = cosine * cos_m - sine * sin_m

# phi2 = cosine * cos_m + sine * sin_m

#phi = torch.where(cosine > th, phi, cosine - mm)
# phi = torch.where(phi   > 0.3, phi, cosine/100)
phi2 = cosine * cos_m - sine * sin_m
lv=np.log(1/(1+np.exp(phi2)))
der=-sine* cos_m - cosine * sin_m
derv=-(1/(1+np.exp(phi2)))*(np.exp(phi2))*der
der2=-cosine * cos_m + sine * sin_m
derv2=(1+np.exp(phi2))*(np.exp(phi2))*der2+(der*np.exp(phi2))**2
acos1=torch.acos(cosine) * 57.29
#phi2=torch.pow(cosine, 3)
#phi2=torch.sigmoid(phi2)

plt.clf()
colors = ['#56B4E9','#E69F00',  '#F0E442', '#009E73', '#D55E00']
fig1,ax1=plt.subplots()
ax1.scatter(acos1,lv, c='r')
ax2=ax1.twinx()

ax2.scatter(acos1,der, c='b')
ax2.scatter(acos1,der2, c='g')


#plt.scatter(acos1,der)

    # Add labels

plt.show()

# output=self.soft(output)