import sys
import utils.objects.dataset as Dataset
sys.path.append('..')
import failure_directions.src.model_utils as model_utils
import failure_directions.src.trainer as trainer_utils
import torch
from torch.cuda.amp import autocast
import torch.nn as nn
from utils import config
import utils.detector.wrappers as wrappers
from abc import abstractmethod
from utils.env import model_env
hparams = config['hparams']

class prototype():
    @abstractmethod
    def __init__(self) -> None:
        self.model = None
    @abstractmethod
    def eval(self, dataloader):
        '''
        gts, preds, probab(not from SVM for now)
        '''
        pass
    @abstractmethod
    def train(self):
        pass
    @abstractmethod
    def save(self, path):
        pass
    @abstractmethod
    def update(self):
        pass
    @abstractmethod
    def load(self, path, device):
        pass
    def acc(self, dataloader):
        gts, preds, _ = self.eval(dataloader)
        return (gts==preds).mean()*100

class resnet(prototype):
    def __init__(self, num_class, use_pretrained=False) -> None:
        super().__init__()
        model_env()
        build_fn = model_utils.BUILD_FUNCTIONS[hparams['arch_type']] # init model factory
        self.model = build_fn(hparams['arch'], num_class, use_pretrained)
        self.model = self.model.cuda() # attach to currect cuda
        self.bce = (num_class==1)

    def load(self, path, device):
        out = torch.load(path, map_location=device)
        self.model.load_state_dict(out['state_dict'])
        print('load model from {}'.format(path))

    def eval(self, dataloader):
        self.model = self.model.eval()
        gts, preds, probabs = [], [], []
        with torch.no_grad():
            with autocast():
                if self.bce:
                    sigmoid = nn.Sigmoid()
                    for batch_info in dataloader:
                        x, y = batch_info[0], batch_info[1]
                        x = x.cuda()
                        gts.append(y.cpu())
                        logits = self.model(x)
                        probab = sigmoid(logits)
                        pred = (probab>=0.5).int()
                        preds.append(pred.cpu())
                        probabs.append(probab.cpu())
                    gts = torch.cat(gts).numpy()
                    preds = torch.cat(preds).numpy()
                    preds = preds.reshape(len(preds))
                    probabs = torch.cat(probabs).numpy()
                    probabs = probabs.reshape(len(probabs))
                else:
                    for batch_info in dataloader:
                        x, y = batch_info[0], batch_info[1]
                        x = x.cuda()
                        gts.append(y.cpu())
                        logits = self.model(x)
                        softmax_logits = nn.Softmax(dim=-1)(logits) # logits: unnormalized output before the last layer
                        # loss.append(ce(softmax_logits,y.cuda()))
                        # probab.append(softmax_logits[torch.arange(logits.shape[0]), y].cpu())
                        probabs.append(softmax_logits.cpu())
                        preds.append(softmax_logits.argmax(-1).cpu())
                    gts = torch.cat(gts).numpy()
                    preds = torch.cat(preds).numpy()
                    probabs = torch.cat(probabs).numpy()
        return gts, preds, probabs            

    def train(self, train_loader, val_loader, log_model=False, model_save_path='', data_weights=None):
        '''return the last checkpoint'''
        training_args=hparams['training']
        training_args['optimizer'] = hparams['optimizer']
        training_args['iters_per_epoch'] = len(train_loader)
        trainer = trainer_utils.LightWeightTrainer(training_args=hparams['training'],
                                                        exp_name=model_save_path, enable_logging=log_model,
                                                        bce=self.bce, set_device=True, loss_upweight_vec=data_weights)
        best_model_chkpnt = trainer.fit(self.model, train_loader, val_loader)
        self.model = best_model_chkpnt

    def tune(self,train_loader,val_loader, weights=None,log_model=False,model_save_path=''):
        '''return the best checkpoint'''
        training_args=hparams['tuning']
        training_args['optimizer'] = hparams['optimizer']
        training_args['iters_per_epoch'] = len(train_loader)
        trainer = trainer_utils.LightWeightTrainer(training_args=training_args,
                                                        exp_name=model_save_path, enable_logging=log_model,
                                                        bce=self.bce, set_device=True,weights=weights)
        best_model_chkpnt = trainer.fit(self.model, train_loader, val_loader)
        self.model = best_model_chkpnt

    def save(self, path):
        torch.save({
            # 'build_params': self.model._build_params,
            'state_dict': self.model.state_dict(),
        }, path)
        print('model saved to {}'.format(path))
    
    def update(self, new_model_setter, train_loader, val_loader):
        if new_model_setter == 'refine':
            self.tune(train_loader,val_loader) # tune
        else:
            self.train(train_loader,val_loader) # retrain 

class svm(prototype):
    def __init__(self, set_up_dataloader, clip_processor:wrappers.CLIPProcessor, split_and_search=False) -> None:
        super().__init__()
        self.clip_processor = clip_processor
        set_up_embedding, _ = self.clip_processor.evaluate_clip_images(set_up_dataloader)        
        self.model = wrappers.SVM(method=config['clf'], svm_args=config['clf_args'],cv=config['clf_args']['k-fold'], split_and_search = split_and_search)
        self.model.set_preprocess(set_up_embedding) 
    
    def load(self, path, device):
        self.model.clf = torch.load(path, map_location=device)

    def eval(self, dataloader):
        fit_embedding, gts = self.clip_processor.evaluate_clip_images(dataloader)
        preds = self.model.base_predict(fit_embedding)
        return gts, preds, None
        
    def train(self, train_loader):
        embedding, gts = self.clip_processor.evaluate_clip_images(train_loader)        
        _ = self.model.base_fit(gts, embedding)

    def save(self, path):
        torch.save(self.model.clf, path)
        print('model saved to {}'.format(path))
    
    def update(self, new_model_config, train_loader, val_loader):
        self.train(train_loader)

def prototype_factory(base_type, cls_num, clip_processor=None):
    if base_type == 'svm':
        return svm(clip_processor)
    elif base_type == 'em':
        return ensembler(cls_num)
    else:
        return resnet(cls_num)
    
import numpy as np
class ensembler(prototype):
    def __init__(self, num_class, use_pretrained=False) -> None:
        self.n_member = 3
        self.n_class = num_class
        self.use_pretrained = use_pretrained
        # self.learner_name = 'resnet'
        self.weights = None
        self.members = []

    def get_weights(self, member_id, dataloader):
        predecessor = self.members[member_id-1]
        gts, preds, _ = predecessor.eval(dataloader)
        if self.weights == None:
            self.weights = np.zeros(len(gts), dtype=bool)
        err_mask = (gts!=preds)
        total_err = err_mask.sum()
        alpha_power = np.power((1 - total_err) / total_err, 0.5)
        new_weights = np.zeros(len(self.weights), dtype=bool)
        new_weights[err_mask] = self.weights[err_mask] * alpha_power
        non_err_mask = ~err_mask
        alpha_power = np.power((1 - total_err) / total_err, -0.5)
        new_weights[non_err_mask] = self.weights[non_err_mask] * alpha_power
        self.weights = new_weights
        self.index_map = self.get_index_map(dataloader)
    
    def get_index_map(self, dataloader):
        index = []
        for batch_info in dataloader:
            batch_index = batch_info[-1]
            index.append(batch_index.cpu())
        return torch.concat(index).numpy()
            
    def train(self, train_loader, val_loader):
        self.base_train(train_loader, val_loader)

    def base_train(self, train_loader, val_loader):
        base = resnet(self.n_class, self.use_pretrained)
        base.train(train_loader, val_loader)
        self.members = [base]

    def update(self, new_model_setter, data_split: Dataset.DataSplits):
        for i in range(1, self.n_member):
            member = resnet(self.n_class, self.use_pretrained)
            self.get_weights(i, data_split.loader['anchor_train'])
            member.train(data_split.loader['train'], data_split.loader['val'], data_weights=self.weights)
            self.members.append(member)

    def eval(self, dataloader):
        if len(self.members) == 1:
            gts, preds, _ = self.members[0].eval(dataloader)
            return gts, preds, None
        else:
            assert len(self.members) == self.n_member
            votes = []
            for member in self.members:
                gts, preds, _ = member.eval(dataloader)
                votes.append(preds)
            votes = np.concatenate(votes, axis=0)
            results = self.get_vote_result(len(preds), votes)
            return gts, results, None

    def load(self, path, device):
        state_dict = torch.load(path, map_location=device)
        for _, member_para in state_dict.items():
            member = resnet(self.n_class, self.use_pretrained)
            member.model.load_state_dict(member_para)
            self.members.append(member)
        assert (len(self.members) == 1) or (len(self.members) == self.n_member), len(self.members)
        print('load model from {}'.format(path))

    def save(self, path):
        state_dict = {}
        if len(self.members) == 1:
            state_dict[0] = self.members[0].model.state_dict()
        else:
            assert len(self.members) == self.n_member
            for idx in range(self.n_member):
                state_dict[idx] = self.members[idx].model.state_dict()
        torch.save(state_dict, path)
        print('model saved to {}'.format(path))

    def get_vote_result(self, dataset_size, votes):
        results = []
        for idx in range(dataset_size):
            vote_cnt = self.count_vote(votes, idx)
            max_id = self.major_vote(vote_cnt)
            results.append(max_id)
        results = np.array(results)
        return results
    
    def count_vote(self, votes, data_idx):
        if self.n_class == 1:
            vote_cnt = {cls_id:0 for cls_id in range(self.n_class + 1)}
        vote_idx = votes[:, data_idx]
        for member_id in range(self.n_member):
            vote_cnt[vote_idx[member_id]] += 1
        return vote_cnt
    
    def major_vote(self, vote_cnt):
        max_vote, max_id = 0, 0
        for cls_id, cls_vote_cnt in vote_cnt.items():
            if cls_vote_cnt > max_vote:
                max_id = cls_id
                max_vote = cls_vote_cnt
        return max_id

