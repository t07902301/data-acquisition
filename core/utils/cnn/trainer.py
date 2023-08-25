import argparse
import yaml
import pprint
from tqdm import tqdm
import torch
import numpy as np
import os
import torch.nn as nn
from copy import deepcopy

from torch.utils.tensorboard import SummaryWriter
from torch.cuda.amp import GradScaler, autocast
from utils.cnn.optimizers import get_optimizer_and_lr_scheduler

def unwrap_batch(batch, set_device=False, bce = False):
    if len(batch) == 4: # cifar
        x, y, fine_y, index = batch
    else: # core
        x, y, obj_y, session, index = batch

    if set_device:
        x = x.cuda()
        y = y.cuda()
    if bce:
        y = y.view(y.shape[0], 1)
    return x, y, index

class AverageMeter():
    def __init__(self):
        self.num = 0
        self.tot = 0
        
    def update(self, val, sz):
        self.num += val*sz
        self.tot += sz
    
    def calculate(self):
        return self.num/self.tot
    
class LightWeightTrainer():
    def __init__(self, training_args, exp_name, enable_logging=True, loss_upweight_vec=None,
                 set_device=False, bce=False, cls_weights=None, loss_weight_index_map=None):
        # print('In Training, loss weights are',weights)
        self.training_args = training_args
        self.bce = bce
        if self.bce:
            self.bce_loss_unreduced = nn.BCEWithLogitsLoss(reduction='none',weight=cls_weights)
            self.bce_loss = nn.BCEWithLogitsLoss(weight=cls_weights)
        else:
            self.ce_loss_unreduced = nn.CrossEntropyLoss(reduction='none',weight=cls_weights)
            self.ce_loss = nn.CrossEntropyLoss(weight=cls_weights)
        self.enable_logging = enable_logging
        if self.enable_logging:
            new_path = self.make_training_dir(exp_name)
            self.training_dir = new_path
            self.writer = SummaryWriter(new_path)
        else:
            self.training_dir = None
            self.writer = None
        self.set_device = set_device
        self.loss_upweight_vec = loss_upweight_vec
        self.loss_weight_index_map = loss_weight_index_map
        # self.sorter = np.argsort(self.loss_weight_index_map) 

    def get_mapped_weights(self, batch_index):
        map_position = []
        for index in batch_index:
            map_position.append(np.where(self.loss_weight_index_map == index)[0])
        map_position = np.array(map_position)
        mapped_weight = self.loss_upweight_vec[map_position]
        return mapped_weight

    # def get_weights(self,labels,cls_num=2):
    #     # cls_num = gts.max() + 1
    #     weights = np.arange(len(labels)) #batch_size
    #     for c in range(cls_num):
    #         mask = labels==c
    #         mask = mask.detach().cpu().numpy()
    #         c_num = np.sum(np.where(mask==True,1,0))
    #         if c_num == 0:
    #             return torch.ones([len(labels)]).cuda()
    #         weights[mask] = 1/c_num
    #         assert (weights >= 0).all().item()

    #     return torch.tensor(weights).cuda() 

    def make_training_dir(self, exp_name):
        path = os.path.join('runs', exp_name)
        os.makedirs(path, exist_ok=True)
        existing_count = -1
        
        for f in os.listdir(path):
            if f.startswith('version_'):
                version_num = f.split('version_')[1]
                if version_num.isdigit() and existing_count < int(version_num):
                    existing_count = int(version_num)
        version_num = existing_count + 1
        new_path = os.path.join(path, f"version_{version_num}")
        print("logging in ", new_path)
        os.makedirs(new_path)
        os.makedirs(os.path.join(new_path, 'checkpoints'))
        return new_path

    def get_accuracy(self, logits, target):
        if self.bce:
            correct = (torch.nn.Sigmoid()(logits) > 0.5).float() == target
        else:
            correct = logits.argmax(-1) == target
        return (correct.float().mean())

    def get_opt_scaler_scheduler(self, model):
        opt, scheduler = get_optimizer_and_lr_scheduler(self.training_args, model)
        scaler = GradScaler()
        return opt, scaler, scheduler

    def training_step(self, model, batch):
        x, y, index = unwrap_batch(batch, self.set_device, self.bce)
        logits = model(x)
        if self.bce:
            temp = self.bce_loss_unreduced(logits, y.float()) # weighted loss
        else:
            temp = self.ce_loss_unreduced(logits, y)
        # get loss weights
        if self.loss_upweight_vec is not None:
            mapped_weight = self.get_mapped_weights(index)
            mapped_weight = mapped_weight.cuda()
            # assert (weight >= 0).all().item()
            temp = temp * mapped_weight

        # weight = self.get_weights(y)
        # assert (weight >= 0).all().item()
        # temp = temp * weight    

        loss = temp.mean()
        acc = self.get_accuracy(logits, y)
        return loss, acc, len(x)
    
    def validation_step(self, model, batch):
        x, y, index = unwrap_batch(batch, self.set_device, self.bce)
        logits = model(x)
        if self.bce:
            loss = self.bce_loss(logits, y.float())
        else:
            loss = self.ce_loss(logits, y)
        acc = self.get_accuracy(logits, y)
        return loss, acc, len(x)
    
    def train_epoch(self, epoch_num, model, train_dataloader, opt, scaler):
        model.train()
        loss_meter = AverageMeter()
        acc_meter = AverageMeter()
        for batch in train_dataloader:
            opt.zero_grad(set_to_none=True)
            with autocast():
                loss, acc, sz = self.training_step(model, batch)
            # t.set_postfix({'loss': loss.item(), 'acc': acc.item()})
            loss_meter.update(loss.item(), sz)
            acc_meter.update(acc.item(), sz)

            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
        avg_loss, avg_acc = loss_meter.calculate(), acc_meter.calculate()
        return avg_loss, avg_acc

    def cyclic_train_epoch(self, epoch_num, model, train_dataloader, opt, scaler, scheduler):
        model.train()
        loss_meter = AverageMeter()
        acc_meter = AverageMeter()
        for batch in train_dataloader:
            opt.zero_grad(set_to_none=True)
            with autocast():
                loss, acc, sz = self.training_step(model, batch)
            # t.set_postfix({'loss': loss.item(), 'acc': acc.item()})
            loss_meter.update(loss.item(), sz)
            acc_meter.update(acc.item(), sz)

            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            scheduler.step()   # TODO Does scheduler update optmizer per batch or per epoch?
               
        # with tqdm(train_dataloader) as t:
        #     t.set_description(f"Train Epoch: {epoch_num}")
        #     for batch in t:
        #         opt.zero_grad(set_to_none=True)
        #         with autocast():
        #             loss, acc, sz = self.training_step(model, batch)
        #         t.set_postfix({'loss': loss.item(), 'acc': acc.item()})
        #         loss_meter.update(loss.item(), sz)
        #         acc_meter.update(acc.item(), sz)

        #         scaler.scale(loss).backward()
        #         scaler.step(opt)
        #         scaler.update()
        #         scheduler.step()
        avg_loss, avg_acc = loss_meter.calculate(), acc_meter.calculate()
        return avg_loss, avg_acc
        
    def val_epoch(self, epoch_num, model, val_dataloader):
        loss_meter = AverageMeter()
        acc_meter = AverageMeter()
        model.eval()
        with torch.no_grad():
            for batch in val_dataloader:
                with autocast():
                    loss, acc, sz = self.validation_step(model, batch)
                loss_meter.update(loss.item(), sz)
                acc_meter.update(acc.item(), sz)            
            # with tqdm(val_dataloader) as t:
            #     t.set_description(f"Val Epoch: {epoch_num}")
            #     for batch in t:
            #         with autocast():
            #             loss, acc, sz = self.validation_step(model, batch)
            #         t.set_postfix({'loss': loss.item(), 'acc': acc.item()})
            #         loss_meter.update(loss.item(), sz)
            #         acc_meter.update(acc.item(), sz)
        avg_loss, avg_acc = loss_meter.calculate(), acc_meter.calculate()
        return avg_loss, avg_acc
    
    def update_model(self, epoch, model, train_dataloader, opt, scaler, scheduler, val_dataloader, lr_scheduler_type):
            if lr_scheduler_type == 'cyclic':
                train_loss, train_acc = self.cyclic_train_epoch(epoch, model, train_dataloader, opt, scaler, scheduler)
                val_loss, val_acc = self.val_epoch(epoch, model, val_dataloader)
            else:
                train_loss, train_acc = self.train_epoch(epoch, model, train_dataloader, opt, scaler)
                val_loss, val_acc = self.val_epoch(epoch, model, val_dataloader)
                if lr_scheduler_type == 'ReduceLROnPlateau': 
                    scheduler.step(val_loss)   
                else:
                    scheduler.step()
            return train_loss, train_acc, val_loss, val_acc

    def fit(self, model, train_dataloader, val_dataloader):
        epochs = self.training_args['epochs']
        opt, scaler, scheduler = self.get_opt_scaler_scheduler(model)
        best_val_loss = np.inf
        best_model_chkpnt = None
        for epoch in range(epochs):
            train_loss, train_acc, val_loss, val_acc = self.update_model(epoch, model, train_dataloader, 
                                                                         opt, scaler, scheduler, val_dataloader, self.training_args['lr_scheduler']['type'])

            # curr_lr = scheduler.get_last_lr()[0]
            # if epoch%10 == 0:
                # print(f"In Epoch: {epoch}, LR: {curr_lr}, Train Loss: {train_loss:0.4f}, Train Acc: {train_acc:0.4f}, Val Loss: {val_loss:0.4f}, Val Acc: {val_acc:0.4f}")
            
            # # Export model with the best checkpoint
            if val_loss < best_val_loss:
                best_val_loss = val_loss   
                del best_model_chkpnt 
                best_model_chkpnt = deepcopy(model)
            # Save Checkpoints
            if self.enable_logging:
                
                self.writer.add_scalar("Loss/train", train_loss, epoch)
                self.writer.add_scalar("Loss/val", val_loss, epoch)
                self.writer.add_scalar("Acc/train", train_acc, epoch)
                self.writer.add_scalar("Acc/val", val_acc, epoch)
                # self.writer.add_scalar("lr", curr_lr, epoch)
                
                run_metadata = {
                    'training_args': self.training_args, 
                    'epoch': epoch, 
                    'training_metrics': {'loss': train_loss, 'acc': train_acc},
                    'val_metrics': {'loss': val_loss, 'acc': val_acc},
                }
                checkpoint_folder = os.path.join(self.training_dir, 'checkpoints')
                checkpoint_path = os.path.join(checkpoint_folder, 'checkpoint_last.pt')
                # save_model(model, checkpoint_path, run_metadata)
                torch.save(run_metadata, checkpoint_path)
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    checkpoint_path = os.path.join(checkpoint_folder, 'checkpoint_best.pt')
                    # save_model(model, checkpoint_path, run_metadata)
                if epoch % 5 == 0: # flush every 5 steps
                    self.writer.flush()
                self.writer.close()
        # print(f"In Epoch: {epoch}, LR: {curr_lr}, Train Loss: {train_loss:0.4f}, Train Acc: {train_acc:0.4f}, Val Loss: {val_loss:0.4f}, Val Acc: {val_acc:0.4f}")
        return best_model_chkpnt