import torch.nn as nn
import torch
import torch.nn.functional as F
import torchaudio
from scipy import signal
from utils import PreEmphasis
from data_loader import FEATURES_LENGTH


class CNN(nn.Module):
    def __init__(self, feature_type, mel, n_mels):
        super(CNN, self).__init__()
        self.mel = mel

        self.fc1 = nn.Sequential(
            nn.Linear(FEATURES_LENGTH[feature_type], 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(),
        )

        self.fc2 = nn.Sequential(
            nn.Linear(512, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(),
        )

        self.fc3 = nn.Sequential(
            nn.Linear(128, 6),
            nn.Softmax(-1)
        )

        print(f'CNN Model init completed, feature_type is {feature_type}, spectrogram is {mel}')

    def forward(self, x1, x, aug=False):

        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)

        return x


if __name__ == '__main__':
    data = torch.randn(32, 1536)
    m = CNN('audeep', 'mfcc', 40)
    o = m.forward(data, data)
    print(o.size())
