import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from nets.nets_utils import weight_init


class Flatten(nn.Module):
    def __init__(self):
        super(Flatten, self).__init__()

    def forward(self, x):
        x = x.transpose(3, 2).contiguous()
        return x.view(x.size(0), -1)

class PNet(nn.Module):
    def __init__(self, is_train=True, use_cuda=False):
        super(PNet, self).__init__()
        self.is_train = is_train
        self.use_cuda = use_cuda

        self.features_layer = nn.Sequential(
            # 12 * 12 *3 ===> 5*5*10
            nn.Conv2d(3, 10, kernel_size=3, stride=1),
            nn.PReLU(10),
            nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True),

            # 5*5*10 ===> 3*3*16
            nn.Conv2d(10, 16, kernel_size=3, stride=1),
            nn.PReLU(16),

            # 3*3*16 ===> 1*1*32
            nn.Conv2d(16, 32, kernel_size=3, stride=1),
            nn.PReLU(32)
        )
        # face classification
        self.conv4_1 = nn.Conv2d(32, 2, kernel_size=1, stride=1)
        #  bounding box regresion
        self.conv4_2 = nn.Conv2d(32, 4, kernel_size=1, stride=1)
        # landmark localization
        self.conv4_3 = nn.Conv2d(32, 10, kernel_size=1, stride=1)
        self.apply(weight_init)

    def forward(self, x):
        x = self.features_layer(x)
        label = F.sigmoid(self.conv4_1(x))
        offset = self.conv4_2(x)
        if self.is_train is True:
            return label, offset
        return label, offset

class RNet(nn.Module):
    def __init__(self, is_train=True, use_cuda=False):
        super(RNet, self).__init__()
        self.is_train = is_train
        self.use_cuda = use_cuda

        self.features_layer = nn.Sequential(

            nn.Conv2d(3, 28, kernel_size=3, stride=1),
            nn.PReLU(28),
            nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True),


            nn.Conv2d(28, 48, kernel_size=3, stride=1),
            nn.PReLU(48),
            nn.MaxPool2d(kernel_size=3, stride=2),

            nn.Conv2d(48, 64, kernel_size=2, stride=1),
            nn.PReLU(64),

            Flatten(),
            nn.Linear(64*3*3, 128),
            nn.PReLU(128)
        )
        # face classification
        self.conv5_1 = nn.Linear(128, 2)
        # bounding box regression
        self.conv5_2 = nn.Linear(128, 4)
        # landmark localization
        self.conv5_3 = nn.Linear(128, 10)
        self.apply(weight_init)

    def forward(self, x):
        x = self.features_layer(x)
        det = F.sigmoid(self.conv5_1(x))
        box = self.conv5_2(x)

        if self.is_train is True:
            return det, box

        return det, box

class ONet(nn.Module):
    def __init__(self, is_train=True, use_cuda=False):
        super(ONet, self).__init__()
        self.is_train = is_train
        self.use_cuda = use_cuda

        self.features_layer = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=1),
            nn.PReLU(32),
            nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True),

            nn.Conv2d(32, 64, kernel_size=3, stride=1),
            nn.PReLU(64),
            nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True),

            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.PReLU(64),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(64, 128, kernel_size=2, stride=1),
            nn.PReLU(128),

            Flatten(),
            nn.Linear(3*3*128, 256),
            nn.PReLU(256)
        )

        # face classification
        self.conv6_1 = nn.Linear(256, 2)
        # bounding box regression
        self.conv6_2 = nn.Linear(256, 4)
        # landmark localization
        self.conv6_3 = nn.Linear(256, 10)
        self.apply(weight_init)

    def forward(self, x):
        x = self.features_layer(x)
        det = F.sigmoid(self.conv6_1(x))
        box = self.conv6_2(x)
        landmark = self.conv6_3(x)

        if self.is_train is True:
            return det, box, landmark

        return det, box, landmark
