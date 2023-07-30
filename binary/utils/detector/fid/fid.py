#Below is to measure FID scores
import numpy as np
from torch.nn.functional import adaptive_avg_pool2d
from pytorch_fid.inception import InceptionV3
import torchvision.transforms as TF

import torch

def get_dataloader_size(dataloader):
    gts = []
    for batch_info in dataloader:
        gts.append(batch_info[1])
    return len(torch.concat(gts))  

def get_inception(dims):

    block_idx = InceptionV3.BLOCK_INDEX_BY_DIM[dims]

    model = InceptionV3([block_idx]).cuda()

    return model

def encode(dataloader, dims=2048):
    model = get_inception(dims)

    loader_size = get_dataloader_size(dataloader)
    pred_arr = torch.empty((loader_size, dims))

    start_idx = 0
    gts = []
    for batch in dataloader:
        img = batch[0].cuda()
        gts.append(batch[1].cpu())

        with torch.no_grad():
            pred = model(img)[0]

        # If model output is not scalar, apply global spatial average pooling.
        # This happens if you choose a dimensionality not equal 2048.
        if pred.size(2) != 1 or pred.size(3) != 1:
            pred = adaptive_avg_pool2d(pred, output_size=(1, 1))

        pred = pred.squeeze(3).squeeze(2)

        pred_arr[start_idx:start_idx + pred.shape[0]] = pred

        start_idx = start_idx + pred.shape[0]

    return pred_arr, torch.cat(gts).numpy()
