# -*- coding: utf-8 -*-

# PyTorch 1.0.1 python 3.7

# =============================================================================
#  @article{zhou2020 IEEE TGRS,
#    title={SAR Image Despeckling Employing a Recursive Deep CNN Prior},
#    author={Shen, Huanfeng and Zhou, Chenxia and Li, Jie and Yuan, QIiangqiang},
#    journal={IEEE Transactions on Remote Sensing},
#    year={2020},
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
import torch
import os
import imageio
from VDRN_ import multi_drn

cuda = torch.cuda.is_available()

os.environ["CUDA_VISIBLE_DEVICES"] = "0"


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--set_names', default='data/test/sentinel1_113_20171021_1220_VH_hill_1_1.tif', help='directory of test dataset')
    parser.add_argument('--model_dir', default=os.path.join('model/', 'SAR_PRCA_'), help='directory of the model')
    parser.add_argument('--model_name', default='model_017.pth', type=str, help='the model name')
    parser.add_argument('--result_dir', default='results/', type=str, help='directory of test dataset')
    parser.add_argument('--iteration', default=2, type=int, help='iteration number for variation')
    parser.add_argument('--la', default=0.55, type=float, help='regularization parameter for variation')
    parser.add_argument('--de', default=0.0005, type=float, help='step lenght for variation')
    return parser.parse_args()


def log(*args, **kwargs):
    print(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S:"), *args, **kwargs)

def min_max_normarlize(img):
    min_img=np.min(img)
    max_img=np.max(img)
    img=(img-min_img)/(max_img-min_img)
    return img, min_img, max_img

#变分模型
def Vatiation(noise,prior,deta,gama):
    x0=noise
    H,W=noise.shape
    x1=(1.0-gama)*x0 +gama*prior- deta * ((x0 - noise) / (x0*x0+0.001) )
    DD=np.sum(np.abs(x1-x0))/(H*W)
    x0=x1
    t=1
    while(DD>0.00001):
        x1 = (1.0-gama)*x0 +gama*prior- deta * ((x0 - noise) / (x0*x0+0.001))
        DD = np.sum(np.abs(x1 - x0)) / (H * W)
        x0 = x1
        t+=1
    return x1

def reback_min_max(img, min_img, max_img):
    img=(img)*(max_img-min_img)+min_img
    return img


if __name__ == '__main__':

    args = parse_args()
    model=multi_drn()
    if cuda:
        model = model.cuda()

    model.load_state_dict(torch.load(os.path.join(args.model_dir, args.model_name)).state_dict())
    log('load trained model')

    model.eval()  # evaluation mode

    path=os.path.join(args.set_names)

    name, ext = os.path.splitext(path)
    name=name.split("\\")
    print(name,name.__len__())

    #影像读取
    y=imageio.imread(path)
    #强度影像转振幅
    y=np.sqrt(y)

    print(y.shape)
    x1_=y
    torch.cuda.synchronize()
    start_time = time.time()
    y_ = x1_ / np.max(x1_)
    z_1 = torch.from_numpy(y_).contiguous().view(-1, 1, y_.shape[0], y_.shape[1])
    z_1 = z_1.cuda()
    with torch.no_grad():
        x1 = model(z_1)  # inference

    x1 = x1.view(y_.shape[0], y_.shape[1])
    x1 = x1.cpu()
    x1 = x1.detach().numpy().astype(np.float32)
    x1 = x1 * np.max(y)
    x1_ = x1
    torch.cuda.synchronize()
    elapsed_time = time.time() - start_time
    print(x1_.shape)
    x1 = np.square(x1_)
    #影像输出为tif
    imageio.imsave(os.path.join(args.result_dir,  name[name.__len__()-1] + '_SAR_DCNN.tif'),x1)

