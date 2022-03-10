import numpy as np
import torch
import random

def print_shape(*a):
    for t in a:
        print(t.shape)
        
def seed_everything(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    random.seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)