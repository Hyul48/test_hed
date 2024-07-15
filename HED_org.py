import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchsummary import summary


class HED(nn.Module):
    def __init__(self):
        super(HED, self).__init__()
        # lr 1 2 decay 1 0
        self.dropout = nn.Dropout(0.2)
        self.conv1_1 = nn.Conv2d(1, 64, 3, padding=1)
        self.conv1_2 = nn.Conv2d(64, 64 , 3, padding=1)

        self.conv2_1 = nn.Conv2d(64, 128, 3, padding=1)
        self.conv2_2 = nn.Conv2d(128, 128, 3, padding=1)

        self.conv3_1 = nn.Conv2d(128, 256, 3, padding=1)
        self.conv3_2 = nn.Conv2d(256, 256, 3, padding=1)
        self.conv3_3 = nn.Conv2d(256, 256, 3, padding=1)

        self.conv4_1 = nn.Conv2d(256, 512, 3, padding=1)
        self.conv4_2 = nn.Conv2d(512, 512, 3, padding=1)
        self.conv4_3 = nn.Conv2d(512, 512, 3, padding=1)

        self.conv5_1 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=2)
        self.conv5_2 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=2)
        self.conv5_3 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=2)
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool2d(2, stride=2)
        self.maxpool4 = nn.MaxPool2d(2, stride=2)

        # lr 0.1 0.2 decay 1 0
        #         self.conv1_1_down = nn.Conv2d(64, 21, 1, padding=0)
        self.score_dsn1 = nn.Conv2d(64, 1, 1, padding=0)

        #         self.conv2_1_down = nn.Conv2d(128, 21, 1, padding=0)
        self.score_dsn2 = nn.Conv2d(128, 1, 1, padding=0)

        #         self.conv3_1_down = nn.Conv2d(256, 21, 1, padding=0)
        #         self.conv3_2_down = nn.Conv2d(256, 21, 1, padding=0)
        self.score_dsn3 = nn.Conv2d(256, 1, 1, padding=0)

        #         self.conv4_1_down = nn.Conv2d(512, 21, 1, padding=0)
        #         self.conv4_2_down = nn.Conv2d(512, 21, 1, padding=0)
        self.score_dsn4 = nn.Conv2d(512, 1, 1, padding=0)

        #         self.conv5_1_down = nn.Conv2d(512, 21, 1, padding=0)
        #         self.conv5_2_down = nn.Conv2d(512, 21, 1, padding=0)
        self.score_dsn5 = nn.Conv2d(512, 1, 1, padding=0)

        self.score_final = torch.nn.Conv2d(in_channels=5, out_channels=1, kernel_size=1, stride=1, padding=0)



    def forward(self, x):
        # VGG
        img_H, img_W = x.shape[2], x.shape[3]
        conv1_1 = self.dropout(self.relu(self.conv1_1(x)))
        conv1_2 = self.dropout(self.relu(self.conv1_2(conv1_1)))
        pool1 = self.maxpool(conv1_2)

        conv2_1 = self.dropout(self.relu(self.conv2_1(pool1)))
        conv2_2 = self.dropout(self.relu(self.conv2_2(conv2_1)))
        #         conv2_2 = self.relu(self.conv2_2(pool1))
        pool2 = self.maxpool(conv2_2)

        conv3_1 = self.dropout(self.relu(self.conv3_1(pool2)))
        conv3_2 = self.dropout(self.relu(self.conv3_2(conv3_1)))
        conv3_3 = self.dropout(self.relu(self.conv3_3(conv3_2)))
        #         conv3_3 = self.relu(self.conv3_3(pool2))
        pool3 = self.maxpool(conv3_3)

        conv4_1 = self.dropout(self.relu(self.conv4_1(pool3)))
        conv4_2 = self.dropout(self.relu(self.conv4_2(conv4_1)))
        conv4_3 = self.dropout(self.relu(self.conv4_3(conv4_2)))
        #         conv4_3 = self.relu(self.conv4_3(pool3))
        pool4 = self.maxpool4(conv4_3)

        conv5_1 = self.dropout(self.relu(self.conv5_1(pool4)))
        conv5_2 = self.dropout(self.relu(self.conv5_2(conv5_1)))
        conv5_3 = self.dropout(self.relu(self.conv5_3(conv5_2)))
        #         conv5_3 = self.relu(self.conv5_3(pool4))

        #         conv1_1_down = self.conv1_1_down(conv1_1)
        tenScoreOne = self.dropout(self.score_dsn1(conv1_2))
        #         conv2_1_down = self.conv2_1_down(conv2_1)
        tenScoreTwo = self.dropout(self.score_dsn2(conv2_2))
        #         conv3_1_down = self.conv3_1_down(conv3_1)
        #         conv3_2_down = self.conv3_2_down(conv3_2)
        tenScoreThr = self.dropout(self.score_dsn3(conv3_3))
        #         conv4_1_down = self.conv4_1_down(conv4_1)
        #         conv4_2_down = self.conv4_2_down(conv4_2)
        tenScoreFou = self.dropout(self.score_dsn4(conv4_3))
        #         conv5_1_down = self.conv5_1_down(conv5_1)
        #         conv5_2_down = self.conv5_2_down(conv5_2)
        tenScoreFiv = self.dropout(self.score_dsn5(conv5_3))
        tenScoreOne = torch.nn.functional.interpolate(input=tenScoreOne,
                                                      size=(x.shape[2], x.shape[3]), mode='bilinear',
                                                      align_corners=False)
        tenScoreTwo = torch.nn.functional.interpolate(input=tenScoreTwo,
                                                      size=(x.shape[2], x.shape[3]), mode='bilinear',
                                                      align_corners=False)
        tenScoreThr = torch.nn.functional.interpolate(input=tenScoreThr,
                                                      size=(x.shape[2], x.shape[3]), mode='bilinear',
                                                      align_corners=False)
        tenScoreFou = torch.nn.functional.interpolate(input=tenScoreFou,
                                                      size=(x.shape[2], x.shape[3]), mode='bilinear',
                                                      align_corners=False)
        tenScoreFiv = torch.nn.functional.interpolate(input=tenScoreFiv,
                                                      size=(x.shape[2], x.shape[3]), mode='bilinear',
                                                      align_corners=False)

        fusecat = torch.cat([tenScoreOne, tenScoreTwo, tenScoreThr, tenScoreFou, tenScoreFiv], 1)
        fuse = self.score_final(fusecat)
        results = [tenScoreOne, tenScoreTwo, tenScoreThr, tenScoreFou, tenScoreFiv, fuse]
        results = [torch.sigmoid(r) for r in results]
        return results

    def crop(variable, th, tw):
        h, w = variable.shape[2], variable.shape[3]
        x1 = int(round((w - tw) / 2.))
        y1 = int(round((h - th) / 2.))
        return variable[:, :, y1: y1 + th, x1: x1 + tw]

    def make_bilinear_weights(size, num_channels):
        factor = (size + 1) // 2
        if size % 2 == 1:
            center = factor - 1
        else:
            center = factor - 0.5
        og = np.ogrid[:size, :size]
        filt = (1 - abs(og[0] - center) / factor) * (1 - abs(og[1] - center) / factor)
        # print(filt)
        filt = torch.from_numpy(filt)
        w = torch.zeros(num_channels, num_channels, size, size)
        w.requires_grad = False
        for i in range(num_channels):
            for j in range(num_channels):
                if i == j:
                    w[i, j] = filt
        return w

