import torch
import numpy as np
import random, os
    
def model_env():
    torch.manual_seed(0)
    torch.cuda.manual_seed(0)
    np.random.seed(0)
    random.seed(0)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ["CUBLAS_WORKSPACE_CONFIG"]=":16:8"

generator = torch.Generator()
generator.manual_seed(0)

def clip_env():
    np.random.seed(0)

def data_split_env():
    np.random.seed(0)
    random.seed(0)    