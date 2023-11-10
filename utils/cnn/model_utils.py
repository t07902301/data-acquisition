import torch
import sys
sys.path.append("..")
import utils.cnn.cifar_resnet as cifar_resnet
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
    return model


def get_cifar_resnet(arch, num_classes):
    cls = {
            'resnet50': cifar_resnet.ResNet50,
            'resnet18': cifar_resnet.ResNet18,
            'resnet34': cifar_resnet.ResNet34,
            'resnet101': cifar_resnet.ResNet101,
            'resnet18wide': cifar_resnet.ResNet18Wide,
            'resnet18thin': cifar_resnet.ResNet18Thin,
    }
    model = cls[arch](num_classes=num_classes)
    model._last_layer_str = 'linear'
    model._build_params = {'arch': arch, 'num_classes': num_classes}
    return model

def get_pretrained(arch, num_classes):
    model = torch.hub.load("chenyaofo/pytorch-cifar-models", arch, pretrained=True)
    # Freeze Feature Extractor
    for param in model.parameters():
        param.requires_grad = False

    if 'resnet' in arch:
        model.fc = nn.Linear(model.fc.in_features, num_classes)
    else:
        model.classifier = nn.Sequential(nn.Dropout(p=0.2, inplace=False),
                        nn.Linear(in_features=model.classifier[1].in_features, out_features=num_classes, bias=True))
    model._build_params = {'arch': arch, 'num_classes': num_classes}
    return model

from utils.cnn.squeezenet import SqueezeNet

def get_raw_model(arch, num_classes):
    if 'resnet' in arch:
        return get_cifar_resnet(arch, num_classes)
    else:
        return SqueezeNet(num_classes)

BUILD_FUNCTIONS = {
    'raw': get_raw_model,
    'pretrained': get_pretrained,
}

