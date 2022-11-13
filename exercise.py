# # import pandas as pd
# # import os
import librosa
import torch
import numpy as np
# # f = os.getcwd().replace("\\","/")
# # feature_path = 'dist/features/'
# # feature_type = 'xbow'
# # sep = ';' if feature_type == 'opensmile' else ','  # 指定分隔符
# # # print("D:/study/网络搭建-python集合/10_AC/ComParE2022/dist/features/xbow/features.csv")
# # # print(f+"/"+feature_path + feature_type + '/features.csv')
# # feat_df = pd.read_csv(feature_path + feature_type + '/features.csv', sep=sep, quotechar="'")  # 加载特征集
#

a =
print(a)
# n_mel = 80
# data1 = np.ones((32,44982))
# res = np.empty((32,80,88))
#
# for i in range(len(data1)):
#     v = data1[i,:]
#     v = v.flatten()
#     v = np.array(data1[i, :])
#     v = v.flatten()
#     stft = np.abs(librosa.stft(v, n_fft=2048, window='hamming'))
#     melW_8k = np.abs(librosa.filters.mel(sr=44100, n_fft=2048, n_mels=80))  # 生成80个梅尔滤波器
#     # np.save('melW_4k.npy',melW_4k)
#     Mel = np.dot(melW_8k, stft)
#     Mel = 20 * np.log(Mel)
#     res[i, :, :] = Mel
# v = np.ones((44982))
# v = v.flatten()
# stft = np.abs(librosa.stft(v, n_fft=2048, window='hamming'))
# melW_8k = np.abs(librosa.filters.mel(sr=44100, n_fft=2048, n_mels=80))  # 生成80个梅尔滤波器
# # np.save('melW_4k.npy',melW_4k)
# Mel = np.dot(melW_8k, stft)
# Mel = 20 * np.log(Mel)
# print(Mel.shape)

# #
# # # a=librosa.feature.mfcc(y=np.array(data1), sr=44100, n_mfcc=n_mel)
# # print(res.shape)
# # print(torch.__version__)
print(torch.cuda.is_available())
# '''
# uar计算
# '''
# # from sklearn.metrics import recall_score
# #
# # pre = [1,2,1,5,8,7,5,6]
# # tru = [1,2,3,6,8,7,5,6]
# #
# # uar = float(recall_score(
# #             tru,
# #             pre,
# #             average="macro",
# #         ))
# #
# # print(uar)
# # a = 1/1 + 1/1 + 0/1 +1 +1/2 + 1 + 1
# # print(a/7)


# b = [4 ,2, 1, 3]
# a = [1,2,3,4]
#
# from collections import deque
# x = []
#
# # x.append(1)
# # x.append(2)
# # print(x[-2:])
#
# for i in range(len(a)):
#
#     x = [ a.pop()] + x
#     if len(x) > 2:
#         x = x[-2:]+x[:-2]
#
# print(x)
# print(b)
#
# # b = b[2:]+b[:2]
# # x.append(b[0])
# # del b[0]
# # print(b)
# #
# # # b = b[2:]+b[:2]
# # # x.append(b[0])
# # # del b[0]
# # # print(b)
# # # print(x)
# #
# # b = b[2:]+b[:2]
# # print(b)
# # b = b[-2:]+b[:-2]
# # # x.append(b[0])
# # # del b[0]
# # print(b)
# # # print(x)
# #



# from collections import Counter

# [n, t] = input().split()
# time_list = input().split()
#
# n = int(n)
# t = int(t)
# time_list = [int(i) for i in time_list]
# n = 6
# t = 5
# time_list = [5 ,6 ,7 ,8 ,9 ,10]
#
# time_list.sort()
# magic_num = 0
#
# c = 0
# s = 0
# for i in range(len(time_list)):
#     while c < time_list[i]:
#         c += t
#         s += 1
# num = len(time_list) - s
#
# print(max(0,num))


# time_interval = [i // t for i in time_list]
# time_counter = Counter(time_interval)
# print(time_counter)
#
# s = 0
# c = 0
# for key, value in time_counter.items():
#     if key == 0:
#         magic_num += value
#     elif key <= n - 1:
#         magic_num += value - 1
# print(magic_num)























