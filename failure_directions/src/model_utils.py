import torch
import sys
sys.path.append("..")
import failure_directions.src.cifar_resnet as cifar_resnet
import torchvision.models as models
import torch.nn as nn


def save_model(model, path, run_metadata):
    torch.save({
        'build_params': model._build_params,
        'state_dict': model.state_dict(),
        'run_metadata': run_metadata
    }, path)

def save_model_svm(model, path, run_metadata):
    torch.save({
        #'build_params': model._build_params,
        'state_dict': model.state_dict(),
        'run_metadata': run_metadata
    }, path)

def load_model(path, build_fn):
    out = torch.load(path)
    model = build_fn(**out['build_params'])
    model.load_state_dict(out['state_dict'])
    # print(out['run_metadata'])
    return model


def get_cifar_resnet(arch, num_classes,use_pretrained=False):
    cls = {
            'resnet50': cifar_resnet.resnet50,
            'resnet18': cifar_resnet.resnet18,
            'resnet34': cifar_resnet.resnet34,
            'resnet101': cifar_resnet.resnet101,
            'resnet18wide': cifar_resnet.resnet18wide,
    }
    model = cls[arch](num_classes=num_classes)
    model._last_layer_str = 'linear'
    model._build_params = {'arch': arch, 'num_classes': num_classes}
    return model

def get_resnet(arch, num_classes, use_pretrained=False):
    resnet_classes = {
        'resnet18': models.resnet18(weights=models.ResNet18_Weights.DEFAULT) if use_pretrained else models.resnet18(weights=None),
        'resnet50': models.resnet50,
    }
    assert arch in resnet_classes
    model = resnet_classes[arch]
    if use_pretrained:
        for param in model.parameters():
            param.requires_grad = False
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    # print(model)
    model._build_params = {'arch': arch, 'num_classes': num_classes}

    # print("Params to learn:")
    # for name,param in model.named_parameters():
    #     if param.requires_grad == True:
    #         print("\t",name)
    # ends

    return model

def get_pretrained_resnet(arch, num_classes):
    return get_resnet(arch, num_classes, pretrained=True)

from failure_directions.src.squeezenet import SqueezeNet

def get_other(arch, num_classes, pretrain=False):
    # import src.other_archs as other_archs

    if arch == 'alexnet':
        return torch.hub.load('pytorch/vision:v0.10.0','alexnet',pretrained=False)
    # elif arch == 'vgg16':
    #     return other_archs.vgg16
    elif arch == 'squeezenet':
        return SqueezeNet(num_classes)
    return None

BUILD_FUNCTIONS = {
    'cifar_resnet': get_cifar_resnet,
    'resnet': get_resnet,
    'pretrained_resnet': get_pretrained_resnet,
    'other': get_other,
}
    
    
