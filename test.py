# Test to check if CUDA is available and print the device being used
import torch
print(torch.cuda.is_available())
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)
import sys
print(sys.path)
