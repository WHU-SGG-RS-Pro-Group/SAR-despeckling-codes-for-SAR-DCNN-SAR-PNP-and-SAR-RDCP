import torch
import torch.nn as nn
from torch.nn import Parameter
import torch.nn.init as init
import torch.nn.functional as F
from torch.autograd import Variable
import cv2
from scipy.io import savemat
T =6

class Recon_(nn.Module):

    def __init__(self, de: float =0.0001, la: float = 0.95):
        super(Recon_, self).__init__()
        self.de = Parameter(data=de*torch.ones(1), requires_grad=False)
        self.la = Parameter(data=la * torch.ones(1), requires_grad=True)
        # self.ga = Parameter(data=ga * torch.ones(1), requires_grad=False)


    def forward(self,  _recon, _feature, _noise):
        _recon = torch.pow(_recon, 2)
        _feature = torch.pow(_feature, 2)
        _noise = torch.pow(_noise, 2)
        # recon = (self.de-self.ga)*_recon-self.ga*_feature+self.la*(torch.div(_recon-_noise, torch.pow(_recon, 2)+1e-3))
        recon =self.la*_recon+(1-self.la)*_feature-self.de*(torch.div(_recon-_noise,torch.pow(_recon, 2)+1e-3))
        return torch.sqrt(recon)

class Recon(nn.Module):
    def __init__(self, de: float =0.001, la: float = 0.55, ga: float = 1.0):
        super(Recon, self).__init__()
        self.de = Parameter(data=de*torch.ones(1), requires_grad=False)
        self.la = Parameter(data=la * torch.ones(1), requires_grad=True)
        self.ga = Parameter(data=ga * torch.ones(1), requires_grad=False)


    def forward(self,  _recon, _feature, _noise):
        _recon = torch.pow(_recon, 2)
        _feature = torch.pow(_feature, 2)
        _noise = torch.pow(_noise, 2)
        # recon = (self.de-self.ga)*_recon-self.ga*_feature+self.la*(torch.div(_recon-_noise, torch.pow(_recon, 2)+1e-3))
        recon = _recon - self.de*(self.la * (torch.div(_recon-_noise, torch.pow(_recon, 2)+1e-3)) + self.ga * (_recon - _feature))
        return torch.sqrt(recon)

class Recon_SO(nn.Module):
    def __init__(self, de: float =0.001, la: float = 0.55, ga: float = 1.0):
        super(Recon_SO, self).__init__()
        self.de = Parameter(data=de*torch.ones(1), requires_grad=False)
        self.la = Parameter(data=la * torch.ones(1), requires_grad=True)
        self.ga = Parameter(data=ga * torch.ones(1), requires_grad=False)

    def forward(self,  _recon, _feature, _noise):
        _recon = torch.log(torch.pow(_recon, 2)+1e-3)
        _feature = torch.pow(_feature, 2)
        _noise = torch.pow(_noise, 2)
        # recon = (self.de-self.ga)*_recon-self.ga*_feature+self.la*(torch.div(_recon-_noise, torch.pow(_recon, 2)+1e-3))
        recon = _recon - self.de*(self.la * (1-_noise*torch.exp(-1*_recon)) + self.ga * (_recon - torch.log(_feature+1e-3)))
        return torch.sqrt(torch.exp(recon))
    # def forward(self,  _recon, _feature, _noise):
    #     recon = _recon - self.de*(-2 * self.la * (torch.div(torch.pow(_noise, 2) - torch.pow(_recon, 2), torch.pow(_recon, 3)+1))
    #                               + self.ga*(_recon - _feature))
    #     return recon
class denoise_block(nn.Module):

    def __init__(self,channels):
        super(denoise_block,self).__init__()
        self.block=nn.Sequential(nn.AvgPool2d(kernel_size=3,stride=1,padding=1),
                                 nn.Conv2d(in_channels=channels,out_channels=channels,kernel_size=1,padding=0))

    def forward(self, x):
        residual=x
        out=self.block(x)
        out+=residual
        return out

class CALayer(nn.Module):
    def __init__(self, channel):
        super(CALayer, self).__init__()
        self.CAbody = nn.Sequential(
            nn.Conv2d(channel, channel, 3, 1, 1, bias=True),
            nn.ReLU(),
            nn.Conv2d(channel, channel, 3, 1, 1, bias=True),
        )
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv_du = nn.Sequential(
                nn.Conv2d(channel, channel, 1, padding=0, bias=True),
                nn.ReLU(inplace=True),
                nn.Conv2d(channel, channel, 1, padding=0, bias=True),
                nn.Sigmoid()
        )

    def forward(self, x):
        out=self.CAbody(x)
        y = self.avg_pool(out)
        y = self.conv_du(y)
        out=out * y
        out+=x
        return out


class DRN_(nn.Module):

    def __init__(self, depth=7, n_channels=64, image_channels=1, kernel_size=3):
        super(DRN_, self).__init__()
        self.net1 = nn.Sequential(
            nn.Conv2d(in_channels=image_channels, out_channels=n_channels, kernel_size=3, padding=1,
                      bias=True, dilation=1), nn.ReLU())
        self.net2 = nn.Sequential(
            nn.Conv2d(in_channels=n_channels, out_channels=n_channels, kernel_size=kernel_size, padding=2,
                      bias=True, dilation=2), nn.ReLU())
        self.net3 = nn.Sequential(
            nn.Conv2d(in_channels=n_channels, out_channels=n_channels, kernel_size=kernel_size, padding=3,
                      bias=True, dilation=3), nn.ReLU())
        self.net4 = nn.Sequential(
            nn.Conv2d(in_channels=n_channels, out_channels=n_channels, kernel_size=kernel_size, padding=4,
                      bias=True, dilation=4), nn.ReLU())
        self.net5 = nn.Sequential(
            nn.Conv2d(in_channels=n_channels, out_channels=n_channels, kernel_size=kernel_size, padding=3,
                      bias=True, dilation=3), nn.ReLU())
        self.net6 = nn.Sequential(
            nn.Conv2d(in_channels=n_channels, out_channels=n_channels, kernel_size=kernel_size, padding=2,
                      bias=True, dilation=2), nn.ReLU())
        self.net7 = nn.Sequential(
            nn.Conv2d(in_channels=n_channels, out_channels=image_channels, kernel_size=3, padding=1,
                      bias=True, dilation=1))
        self._initialize_weights()

    def forward(self, x):
        y = x
        out1 = self.net1(y)
        out2 = self.net2(out1)
        out3 = self.net3(out2)
        out4 = self.net4(out3 + out1)
        out5 = self.net5(out4)
        out6 = self.net6(out5)
        out = self.net7(out6 + out4)
        out = x-out
        # out=self.net8(out)
        return out

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.orthogonal_(m.weight)
                print('init weight')
                if m.bias is not None:
                    init.constant_(m.bias, 0)

class DRN_denoise(nn.Module):

    def __init__(self, depth=7, n_channels=64, image_channels=1, kernel_size=3):
        super(DRN_denoise, self).__init__()
        self.net1 = nn.Sequential(
            nn.Conv2d(in_channels=image_channels, out_channels=n_channels, kernel_size=3, padding=1,
                      bias=True, dilation=1), nn.ReLU())
        self.net2 = nn.Sequential(
            nn.Conv2d(in_channels=n_channels, out_channels=n_channels, kernel_size=kernel_size, padding=2,
                      bias=True, dilation=2), nn.ReLU())
        self.net3 = nn.Sequential(
            nn.Conv2d(in_channels=n_channels, out_channels=n_channels, kernel_size=kernel_size, padding=3,
                      bias=True, dilation=3), nn.ReLU())
        self.net4 = nn.Sequential(
            nn.Conv2d(in_channels=n_channels, out_channels=n_channels, kernel_size=kernel_size, padding=4,
                      bias=True, dilation=4), nn.ReLU())
        self.denoise=denoise_block(channels=n_channels)
        self.net5 = nn.Sequential(
            nn.Conv2d(in_channels=n_channels, out_channels=n_channels, kernel_size=kernel_size, padding=3,
                      bias=True, dilation=3), nn.ReLU())
        self.net6 = nn.Sequential(
            nn.Conv2d(in_channels=n_channels, out_channels=n_channels, kernel_size=kernel_size, padding=2,
                      bias=True, dilation=2), nn.ReLU())
        self.net7 = nn.Sequential(
            nn.Conv2d(in_channels=n_channels, out_channels=image_channels, kernel_size=3, padding=1,
                      bias=True, dilation=1))
        self._initialize_weights()

    def forward(self, x):
        y = x
        out1 = self.net1(y)
        out2 = self.net2(out1)
        out3 = self.net3(out2)
        out4 = self.net4(out3 + out1)
        out4 = self.denoise(out4)
        out5 = self.net5(out4)
        out6 = self.net6(out5)
        out = self.net7(out6 + out4)
        out += x
        # out=self.net8(out)
        return out

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.orthogonal_(m.weight)
                print('init weight')
                if m.bias is not None:
                    init.constant_(m.bias, 0)


class multi_drn(nn.Module):
    def __init__(self, image_channels=1,n_channels=64,kernel_size=3):
        super(multi_drn, self).__init__()
        self.net1 = nn.Sequential(
            nn.Conv2d(in_channels=image_channels, out_channels=n_channels, kernel_size=3, padding=1,
                      bias=True, dilation=1), nn.ReLU())
        self.net2 = nn.Sequential(
            nn.Conv2d(in_channels=n_channels, out_channels=n_channels, kernel_size=kernel_size, padding=2,
                      bias=True, dilation=2), nn.ReLU())
        self.net3 = nn.Sequential(
            nn.Conv2d(in_channels=n_channels, out_channels=n_channels, kernel_size=kernel_size, padding=3,
                      bias=True, dilation=3), nn.ReLU())
        self.net4 = nn.Sequential(
            nn.Conv2d(in_channels=n_channels, out_channels=n_channels, kernel_size=kernel_size, padding=1,
                      bias=True, dilation=1), nn.ReLU())
        self.denoise = denoise_block(channels=int(n_channels))

        self.CAlayer = CALayer(channel=n_channels)
        self.net5 = nn.Sequential(
            nn.Conv2d(in_channels=n_channels, out_channels=n_channels, kernel_size=kernel_size, padding=3,
                      bias=True, dilation=3), nn.ReLU())
        self.net6 = nn.Sequential(
            nn.Conv2d(in_channels=n_channels, out_channels=n_channels, kernel_size=kernel_size, padding=2,
                      bias=True, dilation=2), nn.ReLU())
        self.net7 = nn.Sequential(
            nn.Conv2d(in_channels=n_channels, out_channels=image_channels, kernel_size=3, padding=1,
                      bias=True, dilation=1))
        self._initialize_weights()

    def forward(self, x):
            y = x
            out1 = self.net1(y)
            out2 = self.net2(out1)
            out3 = self.net3(out2)
            out4 = self.net4(out3 + out1)
            out4 = self.denoise(out4)
            out4=self.CAlayer(out4)
            out5 = self.net5(out4)
            out6 = self.net6(out5)
            out = self.net7(out6 + out4)
            out += x
            return out

    def _initialize_weights(self):
            for m in self.modules():
                if isinstance(m, nn.Conv2d):
                    init.orthogonal_(m.weight)
                    print('init weight')
                    if m.bias is not None:
                        init.constant_(m.bias, 0)

class SAR_RDCP(nn.Module):
    def __init__(self,n_channels=64, image_channels=1):
        super(SAR_RDCP,self).__init__()
        self.multi_drn=multi_drn()
        self.recon=Recon(la=0.55,de=0.001)
    def forward(self, x):
        y=x
        out = self.multi_drn(y)
        for i in range(T-1):
            y = self.recon(y, out, x)
            out = self.multi_drn(y)
        # y = self.recon(y, out, x)
        #return out if you want to used SAR-RDCP model, plase change "reture y" to "reture out"
        return out


class switched_SAR_RDCP(nn.Module):
    def __init__(self,n_channels=64, image_channels=1):
        super(switched_SAR_RDCP,self).__init__()
        self.multi_drn=multi_drn()
        self.recon=Recon_(la=0.55,de=0.00055)
    def forward(self, x):
        y=x
        out = self.multi_drn(y)
        for i in range(T-1):
            y = self.recon(y, out, x)
            out = self.multi_drn(y)
        y = self.recon(y, out, x)
        return y



#switched_SAR_RDCP model
class multi_VDRN(nn.Module):
    def __init__(self,n_channels=64, image_channels=1):
        super(multi_VDRN,self).__init__()
        self.multi_drn=multi_drn()
        self.recon=Recon_(la=0.55,de=0.0001)
    def forward(self, x):
        y=x
        out = self.multi_drn(y)
        for i in range(6-1):
            y = self.recon(y, out, x)
            out = self.multi_drn(y)
        y = self.recon(y, out, x)
        #return out if you want to used SAR-RDCP model, plase change "reture y" to "reture out"
        return y
