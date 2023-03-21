# __all__ = ['acquistion', 'dataset', 'model', 'strategy']
import torch
import numpy as np
import random,os
generator = torch.Generator()
generator.manual_seed(0)
torch.manual_seed(0)
torch.cuda.manual_seed(0)
np.random.seed(0)
random.seed(0)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
os.environ["CUBLAS_WORKSPACE_CONFIG"]=":16:8"
import yaml
with open('utils/config.yaml', 'r') as file:
    config = yaml.safe_load(file)
    file.close()