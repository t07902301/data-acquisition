import utils.dataset.wrappers as Dataset
import utils.cnn.model_utils as model_utils
import utils.cnn.trainer as trainer_utils
import torch
from torch.cuda.amp import autocast
import torch.nn as nn
import utils.detector.wrappers as wrappers
from abc import abstractmethod
from utils.env import model_env
import utils.objects.dataloader as dataloader_utils
from utils.logging import logger

class Prototype():
    def __init__(self) -> None:
        self.model = None

    @abstractmethod
    def eval(self, dataloader):
        '''
        gts, preds, probab
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

class CNN(Prototype):
    def __init__(self, hparam_config) -> None:
        super().__init__()
        model_env()
        build_fn = model_utils.BUILD_FUNCTIONS[hparam_config['arch_type']] # init model factory
        self.model = build_fn(hparam_config['arch'], hparam_config['superclass'])
        self.model = self.model.cuda() # attach to current cuda
        self.bce = (hparam_config['superclass']==1)

    def load(self, path, device):
        out = torch.load(path, map_location=device)
        self.model.load_state_dict(out['state_dict'])
        logger.info('load model from {}'.format(path))

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
                    probabs_cls_0 = 1-probabs
                    probabs = probabs.reshape((len(probabs), 1))
                    probabs_cls_0 = probabs_cls_0.reshape((len(probabs_cls_0), 1))
                    probabs = np.concatenate((probabs_cls_0, probabs), axis=1)
                else:
                    for batch_info in dataloader:
                        x, y = batch_info[0], batch_info[1]
                        x = x.cuda()
                        gts.append(y.cpu())
                        logits = self.model(x)
                        softmax_logits = nn.Softmax(dim=-1)(logits) # logits: unnormalized output before the last layer
                        probabs.append(softmax_logits.cpu())
                        preds.append(softmax_logits.argmax(-1).cpu())
                        # loss.append(nn.functional.cross_entropy(logits, y.cuda(), reduction='none').cpu())
                    gts = torch.cat(gts).numpy()
                    preds = torch.cat(preds).numpy()
                    probabs = torch.cat(probabs).numpy()
        return gts, preds, probabs            

    def train(self, train_loader, val_loader, hparam_config, log_model=False, model_save_path='', data_weights=None):
        '''return the best checkpoint'''
        best_val_loss = np.inf
        best_model_chkpnt = None

        training_args=hparam_config['training']
        training_args['optimizer'] = hparam_config['optimizer']
        training_args['iters_per_epoch'] = len(train_loader)
        # logger.info(training_args)
        trainer = trainer_utils.LightWeightTrainer(training_args = training_args,
                                                        exp_name=model_save_path, enable_logging=log_model,
                                                        bce=self.bce, set_device=True, loss_upweight_vec=data_weights)
        best_model_chkpnt, val_loss_check_pnt = trainer.fit(self.model, train_loader, val_loader)

        # epochs, learning_rates = training_args['epochs'], training_args['lr']
        # for epoch in epochs:
        #     for lr in learning_rates:
        #         training_args['epochs'] = epoch
        #         training_args['lr'] = lr
        #         trainer = trainer_utils.LightWeightTrainer(training_args=hparam_config['training'],
        #                                                         exp_name=model_save_path, enable_logging=log_model,
        #                                                         bce=self.bce, set_device=True, loss_upweight_vec=data_weights)
        #         model_check_pnt, val_loss_check_pnt = trainer.fit(self.model, train_loader, val_loader)
        #         if val_loss_check_pnt < best_val_loss:
        #             best_model_chkpnt = model_check_pnt

        self.model = best_model_chkpnt

    def tune(self,train_loader,val_loader, hparam_config, weights=None,log_model=False,model_save_path=''):
        '''return the best checkpoint'''
        training_args=hparam_config['tuning']
        training_args['optimizer'] = hparam_config['optimizer']
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
        logger.info('model saved to {}'.format(path))

class svm(Prototype):
    def __init__(self, hyparams_config, clip_processor:wrappers.CLIPProcessor, split_and_search=True, transform='clip') -> None:
        super().__init__()
        self.transform = transform
        self.clip_processor = clip_processor
        self.model = wrappers.SVM(args=hyparams_config, cv=hyparams_config['k-fold'], split_and_search = split_and_search, do_normalize=False, do_standardize=True)

    def eval(self, dataloader):
        latent, gts = dataloader_utils.get_latent(dataloader, self.clip_processor, self.transform)
        preds = self.model.raw_predict(latent)
        distance, _ = self.model.predict(latent)
        return gts, preds, distance

    def train(self, train_loader):
        latent, gts = dataloader_utils.get_latent(train_loader, self.clip_processor, self.transform)
        self.model.set_preprocess(latent)  #TODO take the mean and std assumed norm dstr
        _ = self.model.fit(latent, gts)

    def save(self, path):
        self.model.export(path)
        logger.info('model saved to {}'.format(path))

    def load(self, path, device):
        self.model.import_model(path)
        logger.info('model load from {}'.format(path))
    
    def update(self, train_loader):
        logger.info('Updating model has train loader of size: {}'.format(dataloader_utils.get_size(train_loader)))

        self.train(train_loader)

class LogReg(Prototype):
    def __init__(self, hyparams_config, clip_processor:wrappers.CLIPProcessor, split_and_search=True, transform='clip') -> None:
        super().__init__()
        self.transform = transform
        self.clip_processor = clip_processor
        self.model = wrappers.LogRegressor(cv=hyparams_config['k-fold'], split_and_search = split_and_search, do_normalize=True, do_standardize=False)

    def eval(self, dataloader):
        latent, gts = dataloader_utils.get_latent(dataloader, self.clip_processor, self.transform)
        preds = self.model.raw_predict(latent)
        distance, _ = self.model.predict(latent)
        return gts, preds, distance

    def train(self, train_loader):
        latent, gts = dataloader_utils.get_latent(train_loader, self.clip_processor, self.transform)
        self.model.set_preprocess(latent)  #TODO take the mean and std assumed norm dstr
        _ = self.model.fit(latent, gts)

    def save(self, path):
        self.model.export(path)
        logger.info('model saved to {}'.format(path))

    def load(self, path, device):
        self.model.import_model(path)
        logger.info('model load from {}'.format(path))
    
    def update(self, train_loader):
        self.train(train_loader)

def factory(model_type, config, clip_processor=None, source=True):
    if model_type == 'svm':
        return svm(config['detector_args'], clip_processor)
    elif source:
        return CNN(config['hparams']['source'])
    else:
        return CNN(config['hparams']['padding'])
    
import numpy as np
class ensembler(Prototype):
    def __init__(self, num_class, use_pretrained=False) -> None:
        self.n_member = 3
        self.n_class = num_class
        self.use_pretrained = use_pretrained
        # self.learner_name = 'CNN'
        self.weights = None
        self.members = []

    def get_weights(self, member_id, dataloader):
        predecessor = self.members[member_id-1]
        gts, preds, _ = predecessor.eval(dataloader)
        if self.weights is None:
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
        base = CNN(self.n_class, self.use_pretrained)
        base.train(train_loader, val_loader)
        self.members = [base]

    def update(self, new_model_setter, data_split: Dataset.DataSplits):
        for i in range(1, self.n_member):
            member = CNN(self.n_class, self.use_pretrained)
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
            member = CNN(self.n_class, self.use_pretrained)
            member.model.load_state_dict(member_para)
            self.members.append(member)
        assert (len(self.members) == 1) or (len(self.members) == self.n_member), len(self.members)
        logger.info('load model from {}'.format(path))

    def save(self, path):
        state_dict = {}
        if len(self.members) == 1:
            state_dict[0] = self.members[0].model.state_dict()
        else:
            assert len(self.members) == self.n_member
            for idx in range(self.n_member):
                state_dict[idx] = self.members[idx].model.state_dict()
        torch.save(state_dict, path)
        logger.info('model saved to {}'.format(path))

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

