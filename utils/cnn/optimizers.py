from torch.optim import SGD, lr_scheduler,Adagrad
import numpy as np

class LRPolicy():
    def __init__(self, lr_schedule):
        self.lr_schedule = lr_schedule
    def __call__(self, epoch):
        return self.lr_schedule[epoch]

def get_optimizer_and_lr_scheduler(training_params, model):
    iters_per_epoch = training_params['iters_per_epoch']
    optimizer_args = training_params['optimizer']
    lr_scheduler_args = training_params['lr_scheduler']
    epochs = training_params['epochs']
    lr = training_params['lr']
    opt_type = optimizer_args['type']

    if opt_type == 'sgd':
        opt = SGD(model.parameters(), 
                lr=lr, 
                momentum=optimizer_args['momentum'],
                weight_decay=optimizer_args['weight_decay'])
    else:   
        opt = Adagrad(model.parameters(),lr= lr,weight_decay=optimizer_args['weight_decay'])

    scheduler_type = lr_scheduler_args['type']
    
    if scheduler_type == 'constant':
        scheduler = None
    elif scheduler_type == 'cyclic':
        lr_peak_epoch = lr_scheduler_args['lr_peak_epoch']
        lr_schedule = np.interp(np.arange((epochs+1) * iters_per_epoch),
                        [0, lr_peak_epoch * iters_per_epoch, epochs * iters_per_epoch],
                        [0, 1, 0])
        scheduler = lr_scheduler.LambdaLR(opt, LRPolicy(lr_schedule))
            
    elif scheduler_type == 'step':
        # scheduler = lr_scheduler.StepLR(opt, step_size=60, 
        #                                 gamma=0.2)
        scheduler = lr_scheduler.MultiStepLR(opt, milestones=[60, 60, 40, 40], gamma=0.2)
    elif scheduler_type == 'exp':
        scheduler = lr_scheduler.ExponentialLR(opt, gamma=0.9)
    else:
        # raise NotImplementedError("Unimplemented LR Scheduler Type")
        scheduler = lr_scheduler.ReduceLROnPlateau(opt, 'min',patience=lr_scheduler_args['patience'])
    return opt, scheduler