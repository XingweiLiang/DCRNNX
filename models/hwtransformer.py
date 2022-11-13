import torch.nn as nn
import torch
from utils import PreEmphasis
import torch.nn.functional as F
from data_loader import FEATURES_LENGTH
# from torch.nn.utils import weight_norm
from models.Layers import FFTBlock

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

def get_padding(kernel_size, dilation=1):
    return int((kernel_size*dilation - dilation)/2)

def get_mask_from_lengths(lengths, max_len=None):
    batch_size = lengths.shape[0]
    if max_len is None:
        max_len = torch.max(lengths).item()

    ids = torch.arange(0, max_len).unsqueeze(0).expand(batch_size, -1).to(device)
    mask = ids >= lengths.expand(-1, max_len)  #lengths.unsqueeze(1).expand(-1, max_len)

    return mask



class HWTrans(nn.Module):

    def __init__(self, feature_type, mel, n_mel):
        super(HWTrans, self).__init__()
        self.feature_type = feature_type
        self.mel = mel
        self.feature_dim = FEATURES_LENGTH[feature_type] if feature_type else 64  # 获取指定特征的维度

        # 通道1 g
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

        self.conv_pre = nn.Conv1d(80, 256, 3, 1, padding=get_padding(3,1))  #n_mel//4*32

        self.layer_stack = nn.ModuleList(

            [
                FFTBlock(
                    d_model=256, n_head=2, d_k=64, d_v=64, d_inner=2048, kernel_size=[9,1], dropout=0.2
                )
                for _ in range(4)
            ]
        )


        self.lstm = nn.LSTM(256,
                            num_layers=2,
                            hidden_size=128,
                            dropout=0.5)

        self.fc = nn.Sequential(
            nn.Linear(256, 64),
            # nn.Linear(256, 64),
        )


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


        # x1 = x1.unsqueeze(1)  # [32, 1, 80, 103]
        # x1 = self.conv1(x1)  # [32, 16, 40, 51]
        # x1 = self.conv2(x1)  # [32, 32, 20, 25]
        # x1 = x1.reshape(x1.shape[0], -1, x1.shape[-1])

        x1 = self.conv_pre(x1)
        max_len = x1.shape[-1]
        batch = x1.shape[0]
        mask = torch.full((batch, max_len), False).to(device)  # None
        slf_attn_mask = mask.unsqueeze(1).expand(-1, max_len, -1)
        x1 = F.leaky_relu(x1, 0.1)
        x1 = x1.permute(0, 2, 1)

        for enc_layer in self.layer_stack:
            x1, enc_slf_attn = enc_layer(x1, mask=mask, slf_attn_mask=slf_attn_mask)


        x1 = x1.permute(1,0,2)
        output, (hc, nc) = self.lstm(x1)
        hc = hc.permute(1, 0, 2)
        x1 = hc.contiguous().view(hc.shape[0], -1)

        x1 = self.fc(x1)

        if self.feature_type:
            # 通道2
            x2 = self.fc2(x2)

            # 拼接倆通道
            x = torch.cat((x1, x2), 1)  # (64, 128)
        else:
            x = x1

        # 分类
        y = self.classification(x)  # (64, 6)

        return y


if __name__ == '__main__':
    # x1 = torch.randn(64, 44100 + 882)
    x1 = torch.randn(64, 80, 88)
    x2 = torch.randn(64, 2000)
    # feature_type = 'opensmile'
    feature_type = 'xbow'
    m = HWTrans(feature_type, 'mfcc', 80)
    y = m.forward(x1, x2)
    print(y.size())