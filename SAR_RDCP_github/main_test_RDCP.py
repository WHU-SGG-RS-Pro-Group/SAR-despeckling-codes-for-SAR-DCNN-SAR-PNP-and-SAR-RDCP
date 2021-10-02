# -*- coding: utf-8 -*-

# PyTorch 1.0.1 python 3.7

# =============================================================================
#  @article{zhou2020 IEEE TGRS,
#    title={SAR Image Despeckling Employing a Recursive Deep CNN Prior},
#    author={Shen, Huanfeng and Zhou, Chenxia and Li, Jie and Yuan, QIiangqiang},
#    journal={IEEE Transactions on Remote Sensing},
#    year={2021},
#    volume={**},
#    number={*},
#    pages={####-####},
#  }
# by Chenxia Zhou (06/2020)
# zhoucx31@gmail.com or zhoucx31@whu.edu.cn
# =============================================================================

# run this to test the model

# ==============================================================================

import argparse
import time, datetime
import numpy as np
import torch.nn as nn
import torch
import os
import imageio
from VDRN_ import multi_VDRN

cuda = torch.cuda.is_available()

os.environ["CUDA_VISIBLE_DEVICES"] = "0"


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--set_names', default='data/test/sentinel1_113_20171021_1220_VH_city_1_1.tif', help='directory of test dataset')
    parser.add_argument('--model_dir', default=os.path.join('model', 'Switched_SAR_RDCP'), help='directory of the model')
    parser.add_argument('--model_name', default='model_020.pth', type=str, help='the model name')
    parser.add_argument('--result_dir', default='results/', type=str, help='directory of test dataset')
    return parser.parse_args()


def log(*args, **kwargs):
    print(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S:"), *args, **kwargs)

def min_max_normarlize(img):
    min_img=np.min(img)
    max_img=np.max(img)
    img=(img-min_img)/(max_img-min_img)
    return img, min_img, max_img


def reback_min_max(img, min_img, max_img):
    img=(img)*(max_img-min_img)+min_img
    return img


if __name__ == '__main__':

    args = parse_args()
    model=multi_VDRN()
    if cuda:
        model = model.cuda()

    model.load_state_dict(torch.load(os.path.join(args.model_dir, args.model_name)).state_dict())
    log('load trained model')

    model.eval()  # evaluation mode
    #变分正则化参数调整
    model.recon.la = nn.Parameter(data=0.26 * torch.ones(1).cuda())



    path=os.path.join(args.set_names)

    name, ext = os.path.splitext(path)
    name=name.split("\\")
    print(name,name.__len__())

    #影像读取
    y=imageio.imread(path)
    #强度影像转振幅
    y[np.where(y >= 13.4)] = 13.4
    print(y.shape)
    y=np.sqrt(y)
    print(y.shape)
    x1_=y[:,:,0]
    torch.cuda.synchronize()
    start_time = time.time()
    y_=x1_/np.max(x1_)
    z_1 = torch.from_numpy(y_).contiguous().view(-1, 1, y_.shape[0], y_.shape[1])
    z_1 = z_1.cuda()
    with torch.no_grad():
        x1 = model(z_1)  # inference

    x1 = x1.view(y_.shape[0], y_.shape[1])
    x1 = x1.cpu()
    x1 = x1.detach().numpy().astype(np.float32)
    x1 =x1 * np.max(y)
    x1=np.square(x1)
    torch.cuda.synchronize()
    elapsed_time = time.time() - start_time
    print(x1_.shape)
    #影像输出为tif
    imageio.imsave(os.path.join(args.result_dir,  name[name.__len__()-1] + '_SAR_RDCP.tif'),x1)

