import torch
import numpy as np
import random, os
    
seed = 0

def model_env():
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ["CUBLAS_WORKSPACE_CONFIG"]=":16:8"

generator = torch.Generator()

def data_env():
    np.random.seed(seed)
    random.seed(seed)    