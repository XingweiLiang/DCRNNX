import torch.nn as nn
import torch
from utils import PreEmphasis
import torch.nn.functional as F
from data_loader import FEATURES_LENGTH
# from torch.nn.utils import weight_norm

def get_padding(kernel_size, dilation=1):
    return int((kernel_size*dilation - dilation)/2)

class ResBlock(torch.nn.Module):
    def __init__(self, in_channels,out_channels, kernel_size=3, dilation=(1, 3, 5)):
        super(ResBlock, self).__init__()

        self.convs1 = nn.ModuleList([
            (nn.Conv1d(in_channels, out_channels, kernel_size, 1, dilation=dilation[0],
                               padding=get_padding(kernel_size, dilation[0]))),
            (nn.Conv1d(out_channels, out_channels, kernel_size, 1, dilation=dilation[1],
                               padding=get_padding(kernel_size, dilation[1]))),
            (nn.Conv1d(out_channels, out_channels, kernel_size, 1, dilation=dilation[2],
                               padding=get_padding(kernel_size, dilation[2])))
        ])

        self.short = nn.Conv1d(in_channels,out_channels,1,1,padding=get_padding(1,1))

        self.c11 = self.convs1[0]
        self.c12 = self.convs1[1]
        self.c13 = self.convs1[2]


        self.convs2 = nn.ModuleList([
            (nn.Conv1d(out_channels, out_channels, kernel_size, 1, dilation=1,
                               padding=get_padding(kernel_size, 1))),
            (nn.Conv1d(out_channels, out_channels, kernel_size, 1, dilation=1,
                               padding=get_padding(kernel_size, 1))),
            (nn.Conv1d(out_channels, out_channels, kernel_size, 1, dilation=1,
                               padding=get_padding(kernel_size, 1)))
        ])


        self.c21 = self.convs2[0]

        self.c22 = self.convs2[1]

        self.c23 = self.convs2[2]



    def forward(self, x):

        xt = F.leaky_relu(x, 0.1)
        xt = self.c11(xt)
        xt = F.leaky_relu(xt, 0.1)
        xt = self.c21(xt)

        if x.shape[1] == xt.shape[1]:
            x = xt + x
        else:
            x = self.short(x) + xt

        xt = F.leaky_relu(x, 0.1)
        xt = self.c12(xt)
        xt = F.leaky_relu(xt, 0.1)
        xt = self.c22(xt)
        x = xt + x

        xt = F.leaky_relu(x, 0.1)
        xt = self.c13(xt)
        xt = F.leaky_relu(xt, 0.1)
        xt = self.c23(xt)
        x = xt + x


        return x


class HWModel(nn.Module):

    def __init__(self, feature_type, mel, n_mel):
        super(HWModel, self).__init__()
        self.feature_type = feature_type
        self.mel = mel
        self.feature_dim = FEATURES_LENGTH[feature_type] if feature_type else 64  # 获取指定特征的维度

        # 通道1 g
        self.conv_pre = nn.Conv1d(80, 128, 3, 1, padding=1)
        resblock = ResBlock


        self.resblock1 = resblock(in_channels=128,out_channels = 64, kernel_size=3, dilation=(1, 3, 5))  #128 64      64  64
        self.resblock2 = resblock(in_channels=64,out_channels = 32, kernel_size=3, dilation=(1, 3, 5))   #64 32      64  32
        self.resblock3 = resblock(in_channels=32,out_channels = 32, kernel_size=3, dilation=(1, 3, 5))   #32 32      32  32

        self.lstm = nn.LSTM(32,
                            num_layers=2,
                            hidden_size=64,
                            dropout=0.5)

        self.fc = nn.Sequential(
            nn.Linear(128, 64),
            # nn.Linear(256, 64),
        )



        # self.conv1 = nn.Sequential(
        #     nn.Conv2d(kernel_size=(3, 3), in_channels=1, out_channels=16, padding=1),
        #     nn.BatchNorm2d(16),
        #     nn.ReLU(),
        #     nn.MaxPool2d(2),
        # )
        #
        # self.conv2 = nn.Sequential(
        #     nn.Conv2d(kernel_size=(3, 3), in_channels=16, out_channels=32, padding=1),
        #     nn.BatchNorm2d(32),
        #     nn.ReLU(),
        #     nn.MaxPool2d(2),
        # )
        #
        # self.lstm = nn.LSTM(32 * (n_mel // 2 // 2),
        #                     num_layers=2,
        #                     hidden_size=128,
        #                     dropout=0.5)
        #
        # self.fc = nn.Sequential(
        #     nn.Linear(256, 64),
        # )

        # 通道2

        # self.fc2_CNN = nn.Sequential(
        #
        #     nn.Conv1d(in_channels=1, out_channels=32, kernel_size=5),
        #     nn.ReLU(),
        #     nn.MaxPool1d(2, stride=1),
        #     nn.BatchNorm1d(32),
        #
        #     nn.Conv1d(in_channels=32, out_channels=64, kernel_size=7),
        #     nn.ReLU(),
        #     nn.MaxPool1d(2, stride=1),
        #     nn.BatchNorm1d(64),
        #
        #     nn.Conv1d(in_channels=64, out_channels=64, kernel_size=9),
        #     nn.ReLU(),
        #     nn.MaxPool1d(2, stride=1),
        #     nn.BatchNorm1d(64),
        #     nn.AdaptiveAvgPool1d(1),
        #
        # )

        # self.f2l = nn.Sequential(
        #     nn.Linear(128, 64),
        #     nn.BatchNorm1d(64),
        #     nn.ReLU(),
        #     nn.Dropout(),
        # )



        self.fc2 = nn.Sequential(
            nn.Linear(FEATURES_LENGTH[feature_type], 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(),

            nn.Linear(512, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(),
        )

        in_features = 128 if feature_type else 64

        self.classification = nn.Sequential(
            nn.Linear(in_features=in_features, out_features=6),
            nn.Softmax(dim=-1)
        )

        print(f'TwoChannel Model init completed, feature_type is {feature_type}, spectrogram is {mel}, '
              f'n_mel is {n_mel}')

    def forward(self, x1, x2, aug=True):
        # 通道1

        x1 = self.conv_pre(x1)
        x1 = F.leaky_relu(x1, 0.1)

        # xs = self.resblock1(x1)
        # xs += self.resblock2(x1)
        # xs += self.resblock3(x1)
        #
        # x1 = xs / 3
        #
        # x1 = F.leaky_relu(x1, 0.1)

        xs = self.resblock1(x1)

        xs = self.resblock2(xs)

        xs = self.resblock3(xs)

        x1 = xs.permute(2,0,1)
        output, (hc, nc) = self.lstm(x1)
        hc = hc.permute(1, 0, 2)
        x1 = hc.contiguous().view(hc.shape[0], -1)
        x1 = self.fc(x1)


        # x1 = x1.unsqueeze(1)  # [32, 1, 80, 103]
        # x1 = self.conv1(x1)  # [32, 16, 40, 51]
        # x1 = self.conv2(x1)  # [32, 32, 20, 25]
        #
        # x1 = x1.reshape(x1.shape[0], -1, x1.shape[-1]).permute(2, 0, 1)  # [25, 32, 32 * 20]
        # output, (hc, nc) = self.lstm(x1)  # [25, 32, 128], [2, 32, 128]
        #
        # hc = hc.permute(1, 0, 2)  # [32, 2, 128]
        # hc = hc.contiguous().view(hc.shape[0], -1)  # [32, 256]
        # x1 = self.fc(hc)  # [32, 64]

        if self.feature_type:
            # 通道2
            # x2 = x2.unsqueeze(1)
            # x2 = self.fc2(x2)
            # x2 = x2.squeeze(-1)
            # x2 = self.f2l(x2)



            x2 = self.fc2(x2)
            # x2 = x2.unsqueeze(-1)

            # x2 = x2.squeeze(-1)
            # x2 = self.fc3(x2)

            # 拼接倆通道
            x = torch.cat((x1, x2), 1)  # (64, 128)
        else:
            x = x1

        # 分类
        y = self.classification(x)  # (64, 6)

        return y


if __name__ == '__main__':
    # x1 = torch.randn(64, 44100 + 882)
    x1 = torch.randn(64, 40, 88)
    x2 = torch.randn(64, 2000)
    # feature_type = 'opensmile'
    feature_type = 'xbow'
    m = HWModel(feature_type, 'mfcc', 40)
    y = m.forward(x1, x2)
    print(y.size())