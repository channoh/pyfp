
import torch
import torch.nn.functional as F
import custom_fp as fp

torch.manual_seed(0)

pad_h = 0
pad_w = 0

stride_h = 1
stride_w = 1

ifm_c = 1024
ifm_h = 100
ifm_w = 100

wgt_h = 3
wgt_w = 3

ofm_c = 128
ofm_h = (ifm_h - wgt_h + pad_h * 2) // stride_h + 1
ofm_w = (ifm_w - wgt_w + pad_w * 2) // stride_w + 1

dilation = 1

ifm = torch.FloatTensor(1, ifm_c, ifm_h, ifm_w).normal_(mean=0, std=1)
wgt = torch.FloatTensor(ofm_c, ifm_c, wgt_h, wgt_w).normal_(mean=0, std=1)
bias = torch.zeros(ofm_c)
ofm = torch.zeros(1, ofm_c, ofm_h, ofm_w)

fp.my_conv2d(ofm, ifm, wgt, bias, stride_h, stride_w, pad_h, pad_w, dilation)
print(ofm[0][0][0][0])


ref = F.conv2d(ifm, wgt, bias, stride_h, pad_h, dilation)
print(ref[0][0][0][0])


