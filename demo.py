# -*- coding: gbk -*-
# @Time    : 2024-04-02 21:08
# @Author  : Yao Cui
# @E-mail  : cy_cumtb@hotmail.com
import warnings
import numpy as np
from torchvision import transforms
from spectral import *
warnings.filterwarnings("ignore")


# hdr_file = r"E:\spectral_data\powder\Exploration_experiment_condition\20240402\High_Low_Coal_Ash_4\REFLECTANCE_SWIR_emptyname_2024-04-02_02-05-44.hdr"
# dat_file = r"E:\spectral_data\powder\Exploration_experiment_condition\20240402\High_Low_Coal_Ash_4\REFLECTANCE_SWIR_emptyname_2024-04-02_02-05-44.dat"

hdr_file = r"E:\spectral_data\powder\Exploration_experiment_condition\20240402\Classical Mineral_3_Concrete_7\spectral_image\REFLECTANCE_SWIR_emptyname_2024-04-02_03-23-18.hdr"
dat_file = r"E:\spectral_data\powder\Exploration_experiment_condition\20240402\Classical Mineral_3_Concrete_7\spectral_image\REFLECTANCE_SWIR_emptyname_2024-04-02_03-23-18.dat"

# hdr_file = r"E:\spectral_data\powder\Exploration_experiment_condition\20240402\Pure_4\spectral_image\REFLECTANCE_SWIR_emptyname_2024-04-02_02-32-53.hdr"
# dat_file = r"E:\spectral_data\powder\Exploration_experiment_condition\20240402\Pure_4\spectral_image\REFLECTANCE_SWIR_emptyname_2024-04-02_02-32-53.dat"
# concrete_coal = envi.open(hdr_file, dat_file).asarray()[34:79, 119:164, :]  # 119 34 164 79
# concrete_coal = envi.open(hdr_file, dat_file).asarray()[153:198, 115:160, :]  # 115 153 160 198
# concrete_coal = envi.open(hdr_file, dat_file).asarray()[507:552, 123:168, :]  # 123 507 168 552

concrete_coal = envi.open(hdr_file, dat_file).asarray()[394:439, 219:264, :]  # 219 394 264 439
# print('---concrete_coal', type(concrete_coal))
# concrete_coal = envi.open(hdr_file, dat_file).asarray()[80:125, 78:123, :]  # 78 80 123 125
# concrete_coal_1 = envi.open(hdr_file, dat_file).asarray()[199:244, 83:128, :]  # 83 199 128 244
# concrete_coal_2 = envi.open(hdr_file, dat_file).asarray()[77:122, 195:240, :]  # 195 77 240 122
# concrete_coal_3 = envi.open(hdr_file, dat_file).asarray()[196:241, 193:238, :]  # 193 196 238 241

view_cube(concrete_coal)
imshow(concrete_coal)
imshow(concrete_coal_1)
imshow(concrete_coal_2)
imshow(concrete_coal_3)
# print('---data', concrete_coal.shape)  # (306, 384, 272)
view_cube(concrete_coal)

npy_file_dir = r"E:\spectral_data\powder\Spectral-Data-ZKH-Sample-Expansion\ppt_8.5728_6.37e-5_1_000_0.npy"
npy_file = np.load(file=npy_file_dir).astype(np.float32)
npy_file = np.squeeze(npy_file)
npy_file_1 = npy_file
npy_file = (npy_file * 255).astype(np.uint8)
npy_file = npy_file.transpose(2,1,0)
print('----npy_file', npy_file[:3,:,:])  # 100 * 100 * 3
print('----npy_file_1', npy_file_1[:3,:,:])  # 100 * 100 * 3
#
transform = transforms.ToTensor()
tensor_hs_image = transform(npy_file)  # 从H x W x B转换为B x H x W
print('----tensor_hs_image', tensor_hs_image)  # 100 * 100 * 3
# print(tensor_hs_image.shape)  # 应输出：torch.Size([215, 100, 100])

import torch
from torchvision import transforms

# 模拟一个高光谱图像数据（例如100波段，100x100像素）


# hs_image = np.random.randint(1, 10, size=[3, 5, 5]).astype(np.float32)
# print("转换前", hs_image.shape, hs_image)  # 应输出：torch.Size([100, 100, 100])
#
# # 转换为PyTorch张量
# transform = transforms.ToTensor()
# tensor_hs_image = transform(hs_image)  # 从H x W x B转换为B x H x W
#
# print("转换后", tensor_hs_image.shape, tensor_hs_image)  # 应输出：torch.Size([100, 100, 100])
