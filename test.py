# Test to check if CUDA is available and print the device being used
# import torch
# print(torch.cuda.is_available())
# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# print(device)
# import sys
# print(sys.path)

from itertools import chain
import numpy as np
data=[[[0,1,4],[2,3,4,5]],[[6,7,8],[9,10,11,12]]]
data=np.array(data,dtype=object)
print(data)
share_obs = []
for o in data:
    share_obs.append(list(chain(*o)))
share_obs = np.array(share_obs)
print(share_obs)
data = np.array(list(data[:,0])).copy()
print(data)