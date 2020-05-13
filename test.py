
import torch
import custom_fp as fp


val = torch.tensor([1])
ret = torch.tensor([1])
fp.inc(ret, val)
print(ret)
