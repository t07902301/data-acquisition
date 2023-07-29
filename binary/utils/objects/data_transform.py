from utils.detector.wrappers import CLIPProcessor
import torch

def count_minority(ds):
    '''
    return #minority in ds
    '''
    minority_labels = [4, 73, 54, 10, 51, 40, 84, 18, 3, 12, 33, 38, 64, 45, 2, 44, 80, 96, 13, 81]
    cnt = 0
    ds_size = len(ds)
    for index in range(ds_size):
        if ds[index][2] in minority_labels:
            cnt += 1
    return cnt

def loader2dataset(dataloader):
    img, coarse_labels, fine_labels = [], [], []
    for batch_info in dataloader:
        x, y, fine_y, _  = batch_info
        img.append(x)
        coarse_labels.append(y)
        fine_labels.append(fine_y)
    img = torch.cat(img)
    coarse_labels = torch.cat(coarse_labels)
    fine_labels = torch.cat(fine_labels)

    data = []
    for i in range(len(img)):
        data.append((img[i], coarse_labels[i], fine_labels[i]))
    return data

def get_dataloader_labels(dataloader):
    gts = []
    for batch_info in dataloader:
        gts.append(batch_info[1].cpu())
    return torch.concat(gts).numpy()

def get_dataloader_size(dataloader):
    gts = []
    for batch_info in dataloader:
        gts.append(batch_info[1])
    return len(torch.concat(gts))  

def get_flattened(loader):
    img = []
    for batch_info in loader:
        img.append(torch.flatten(batch_info[0], start_dim=1))
    return torch.cat(img, dim=0)

def get_latent(data_loader, clip_processor:CLIPProcessor = None, transform: str = None):
    if transform == 'clip':
        latent, _ = clip_processor.evaluate_clip_images(data_loader)  
    elif transform == 'flatten':
        latent = get_flattened(data_loader) 
    else:
        latent = loader2data(data_loader)
    return latent

def loader2data(loader):
    img = []
    for batch_info in loader:
        img.append(batch_info[0])
    return torch.cat(img, dim=0)