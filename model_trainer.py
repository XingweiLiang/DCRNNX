import time
import os

import librosa
import soundfile
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import recall_score
import numpy as np
import random

from data_loader import EMOTION_MAPPING
from models.two_channel import TwoChannel
from models.three_channel import ThreeChannel
from models.crnn import CRNN
from models.cnn import CNN
from models.lstm import LSTM
from models.crnn2 import CRNN2
# from models.tmp import CNN_2
from models.attention_base import AttentionBase
from models.crnn3 import CRNN3
from models.hwmodel import HWModel
from models.hwtransformer import HWTrans


device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
# device = torch.device('cpu')


MODEL_MAPPING = {
    'crnn': CRNN,
    'cnn': CNN,
    'lstm': LSTM,
    'twochannel': TwoChannel,
    'threechannel': ThreeChannel,
    'crnn2': CRNN2,
    # 'tmp': CNN_2,
    'attbase': AttentionBase,
    'crnn3': CRNN3,
    'hwmodel': HWModel,
    'hwtransformer':HWTrans,
}


class ModelTrainer(nn.Module):

    def __init__(self, lr, test_step, lr_decay, feature_type, model, mel, n_mel, **kwargs):
        super(ModelTrainer, self).__init__()
        # 定义模型
        self.model = MODEL_MAPPING[model](feature_type, mel ,n_mel).to(device)
        # 定义损失函数
        self.loss = nn.CrossEntropyLoss().to(device)
        # 定义优化器
        self.optim = torch.optim.Adam(self.parameters(), lr=lr, weight_decay=2e-5)
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optim, step_size=test_step, gamma=lr_decay)

        # 打印模型参数大小
        print(time.strftime("%m-%d %H:%M:%S") + " Model para number = %.2f" % (
                sum(param.numel() for param in self.model.parameters()) / 1024 / 1024))

    def train_network(self, epoch, loader, mel, n_mel):
        self.train()
        # Update the learning rate based on the current epcoh
        self.scheduler.step(epoch - 1)
        lr = self.optim.param_groups[0]['lr']
        total_loss = 0

        for num, (data1, data2, labels) in enumerate(loader, start=1):
            self.zero_grad()  # 当前梯度置为0
            labels = torch.LongTensor(labels)
            labels = labels.to(device)

            res = np.empty((len(data1), 80, 88))
            if mel == 'mfcc':
                #Mel谱
                for i in range(len(data1)):
                    v = np.array(data1[i, :])
                    v = v.flatten()
                    stft = np.abs(librosa.stft(v, n_fft=2048, window='hamming'))
                    melW_8k = np.abs(librosa.filters.mel(sr=44100, n_fft=2048, n_mels=80))  # 生成80个梅尔滤波器
                    # np.save('melW_4k.npy',melW_4k)
                    Mel = np.dot(melW_8k, stft)
                    Mel = 20 * np.log(Mel)
                    res[i, :, :] = Mel

                data1 = torch.FloatTensor(res)


                # 原始代码重写mfcc
                # for i in range(len(data1)):
                #     v = np.array(data1[i, :])
                #     v = v.flatten()
                #     res[i, :, :] = librosa.feature.mfcc(y=v, sr=44100, n_mfcc=n_mel)
                #
                # data1 = torch.FloatTensor(res)

                #原始代码
                # data1 = torch.FloatTensor(librosa.feature.mfcc(y=np.array(data1), sr=44100, n_mfcc=n_mel))

            data1 = data1.to(device)
            data2 = data2.to(device)
            res_embedding = self.model.forward(data1, data2)

            loss = self.loss.forward(res_embedding, labels)
            loss.backward()
            self.optim.step()
            total_loss += loss

        return lr, total_loss

    def eval_network(self, devel_path, wav_path, feature_type, feat_df, mel, n_mel):
        self.eval()

        # 加载label
        label = 'dist/lab/test.csv'
        label_df = pd.read_csv(devel_path, sep=',')

        # 加载指定特征
        if feature_type:
            name = 'filename' if feature_type == 'audeep' else 'name'
            devel_df = feat_df[feat_df[name].str.contains('devel')].reset_index(drop=True)
            # devel_df = feat_df[feat_df[name].str.contains('test')].reset_index(drop=True)
        # label to emotion
        l2e = {v: k for k, v in EMOTION_MAPPING.items()}

        # 预测结果，实际结果
        pred_list, true_list = [], []

        with torch.no_grad():
            for index in range(len(label_df)):
                # 原始语音
                audio, sr = soundfile.read(os.path.join(wav_path, label_df.loc[index, 'filename']))
                # audio = librosa.effects.pitch_shift(audio, sr, n_steps=3)

                # length = sr * 1 + 882  # sr * 1: 1s的语音长度，882：是30ms一帧，10ms一帧移，最后一帧会少20ms的语音长度
                # if audio.shape[0] <= length:
                #     shortage = length - audio.shape[0]
                #     audio = np.pad(audio, (0, shortage), 'wrap')
                # start_frame = np.int64(random.random() * (audio.shape[0] - length))
                # audio = audio[start_frame: start_frame + length]
                audio /= np.std(audio)
                audio -= np.mean(audio)

                audio = torch.FloatTensor(audio).unsqueeze(0)
                if mel == 'mfcc':
                    # audio = torch.FloatTensor(librosa.feature.mfcc(y=np.array(audio).flatten(), sr=44100, n_mfcc=n_mel))
                    stft = np.abs(librosa.stft(np.array(audio).flatten(), n_fft=2048, window='hamming'))
                    melW_8k = np.abs(librosa.filters.mel(sr=44100, n_fft=2048, n_mels=80))  # 生成80个梅尔滤波器
                    # np.save('melW_4k.npy',melW_4k)
                    Mel = np.dot(melW_8k, stft)
                    Mel = 20 * np.log(Mel)
                    audio = torch.FloatTensor(Mel)
                    audio = audio.unsqueeze(0)
                data1 = audio.to(device)

                # 指定特征
                if feature_type:
                    devel_feat = devel_df[devel_df[name] == label_df.loc[index, 'filename']]
                    devel_feat = devel_feat.iloc[0, 1:].astype(float).values
                    devel_feat = torch.FloatTensor(devel_feat).unsqueeze(0)
                    data2 = devel_feat.to(device)
                else:
                    data2 = 0.0

                output = self.model.forward(data1, data2, aug=False)
                pred_list.append(l2e[output.argmax().item()])
                true_list.append(label_df.loc[index, 'label'])

        # 计算UAR(未加权平均召回)
        uar = float(recall_score(        #平均召回率，左边为真实标签，上方为预测标签统计混淆矩阵，计算每一类真实标签中预测对的概率，再对其取平均得到总体的uar
            true_list,
            pred_list,
            average="macro",
        ))

        return uar, true_list, pred_list

    def eval_network_n(self, devel_path, wav_path, feature_type, feat_df, mel, n_mel):
        self.eval()

        # 加载label
        label = 'dist/lab/test.csv'
        label_df = pd.read_csv(devel_path, sep=',')

        # 加载指定特征
        if feature_type:
            name = 'filename' if feature_type == 'audeep' else 'name'
            devel_df = feat_df[feat_df[name].str.contains('devel')].reset_index(drop=True)
            # devel_df = feat_df[feat_df[name].str.contains('test')].reset_index(drop=True)
        # label to emotion
        l2e = {v: k for k, v in EMOTION_MAPPING.items()}

        # 预测结果，实际结果
        pred_list, true_list = [], []
        output_list = []

        with torch.no_grad():
            for index in range(len(label_df)):
                # 原始语音
                audio, sr = soundfile.read(os.path.join(wav_path, label_df.loc[index, 'filename']))

                # length = sr * 1 + 882  # sr * 1: 1s的语音长度，882：是30ms一帧，10ms一帧移，最后一帧会少20ms的语音长度
                # if audio.shape[0] <= length:
                #     shortage = length - audio.shape[0]
                #     audio = np.pad(audio, (0, shortage), 'wrap')
                # start_frame = np.int64(random.random() * (audio.shape[0] - length))
                # audio = audio[start_frame: start_frame + length]
                audio /= np.std(audio)
                audio -= np.mean(audio)

                audio = torch.FloatTensor(audio).unsqueeze(0)
                if mel == 'mfcc':
                    audio = torch.FloatTensor(librosa.feature.mfcc(y=np.array(audio), sr=44100, n_mfcc=n_mel))
                data1 = audio.to(device)

                # 指定特征
                if feature_type:
                    devel_feat = devel_df[devel_df[name] == label_df.loc[index, 'filename']]
                    devel_feat = devel_feat.iloc[0, 1:].astype(float).values
                    devel_feat = torch.FloatTensor(devel_feat).unsqueeze(0)
                    data2 = devel_feat.to(device)
                else:
                    data2 = 0.0

                output = self.model.forward(data1, data2, aug=False)
                output_list.append(np.array(output.cpu())[0])
                pred_list.append(l2e[output.argmax().item()])
                true_list.append(label_df.loc[index, 'label'])

        # 计算UAR(未加权平均召回)
        uar = float(recall_score(
            true_list,
            pred_list,
            average="macro",
        ))
        df = pd.DataFrame(output_list)
        df.to_excel('model_pred.xlsx')
        return uar, true_list, pred_list

    def save_parameters(self, path):
        torch.save(self.state_dict(), path)

    def load_parameters(self, path):
        self_state = self.state_dict()
        loaded_state = torch.load(path)
        for name, param in loaded_state.items():
            origname = name
            if name not in self_state:
                name = name.replace("module.", "")
                if name not in self_state:
                    print("%s is not in the model." % origname)
                    continue
            if self_state[name].size() != loaded_state[origname].size():
                print("Wrong parameter length: %s, model: %s, loaded: %s" % (
                origname, self_state[name].size(), loaded_state[origname].size()))
                continue
            self_state[name].copy_(param)