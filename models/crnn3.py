import torch.nn as nn
import torch
import torch.nn.functional as F
import torchaudio
from scipy import signal
from utils import PreEmphasis
from data_loader import FEATURES_LENGTH
import librosa
import numpy as np


class CRNN3(nn.Module):
    def __init__(self, feature_type, mel, n_mels):
        super(CRNN3, self).__init__()
        self.feature_type = feature_type
        self.mel = mel

        # fbank特征
        self.torchfbank = torch.nn.Sequential(
            PreEmphasis(),
            torchaudio.transforms.MelSpectrogram(sample_rate=44100, n_fft=2048, win_length=1323, hop_length=441, \
                                                 f_min=0, f_max=44100 // 2, window_fn=torch.hamming_window,
                                                 n_mels=n_mels)
        )

        in_channels = FEATURES_LENGTH[feature_type] if feature_type else n_mels
        self.conv1 = nn.Sequential(
            nn.Conv2d(kernel_size=(3, 3), in_channels=1, out_channels=32, padding=(1, 1)),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(kernel_size=(5, 5), in_channels=1, out_channels=32, padding=(2, 2)),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )

        self.conv3 = nn.Sequential(
            nn.Conv2d(kernel_size=(3, 3), in_channels=64, out_channels=128, padding=(1, 1)),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            # nn.MaxPool2d(2),
        )

        self.lstm = nn.LSTM(128 * (n_mels // 2), num_layers=2, hidden_size=128)

        self.fc = nn.Sequential(
            nn.Linear(256, 64),
            nn.Linear(64, 6),
            nn.Softmax(dim=1),
        )

        print(f'CRNN3 Model init completed, feature_type is {feature_type}, spectrogram is {mel}, '
              f'n_mel is {n_mels}')

    def forward(self, x1, x2, aug=False):
        x = x1
        if self.mel == 'fbank':
            with torch.no_grad():
                x1 = self.torchfbank(x1) + 1e-6
                x1 = x1.log()
                x = x1 - torch.mean(x1, dim=-1, keepdim=True)

        x = x.unsqueeze(1)  # [32, 1, 40, 88]
        x1 = self.conv1(x)  # [32, 32, 20, 44]
        x2 = self.conv2(x)  # [32, 32, 20, 44]

        x = torch.cat((x1, x2), dim=1)      # [32, 64, 20, 44]
        x = self.conv3(x)   # [32, 128, 20, 44]
        # x = x.squeeze(1).permute(2, 0, 1)    # [32, 20, 44] -> [44, 32, 20]

        x = x.reshape(x.shape[0], -1, x.shape[-1]).permute(2, 0, 1)     # [44, 32, 128 * 20]
        output, (hc, nc) = self.lstm(x)  # [44, 32, 128], [2, 32, 128]

        hc = hc.permute(1, 0, 2)    # [32, 2, 128]
        hc = hc.contiguous().view(hc.shape[0], -1)   # [32, 256]
        y = self.fc(hc)   # [32, 6]

        return y


if __name__ == '__main__':
    # data = torch.randn(32, 44100 + 882)
    data = torch.randn(32, 40, 88)
    m = CRNN3(None, 'mfcc', 40)
    o = m.forward(data, data)
    print(o.size())
