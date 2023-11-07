import time
import numpy as np
import torch

def run(a, b):
    c = torch.mm(a, b)
    # c = np.dot(a, b)

if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')

# a = torch.zeros((100, 100)).to(device)
# b = torch.zeros((100, 100)).to(device)

a = np.zeros((100, 100))
b = np.zeros((100, 100))

start = time.time()
run(a, b)
end = time.time()
elapsed_time = end - start
print("elapsed_time:{0}".format(elapsed_time) + "[sec]")