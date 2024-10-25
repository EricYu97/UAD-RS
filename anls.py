import torch
import torch.nn as nn
from torch.utils.data import DataLoader,Dataset
from torchvision.datasets import ImageFolder
import os
import models_cls, models_seg
from torchvision.models._utils import IntermediateLayerGetter
import piqa
import argparse
from collections import OrderedDict
from datasets import load_dataset
import torchvision.transforms as tfs

def args():
    parser = argparse.ArgumentParser(description='ANLS Scheme')
    parser.add_argument('--dataset', type=str, default='AID', help='dataset name, choices: [AID,UCM,Vaihingen,Zurich]')
    parser.add_argument('--root_path', type=str, default='/home/yu34/AdvD/UAE-RS', help='root path of UAE-RS')
    parser.add_argument('--model', type=str, default='alexnet', help='model name')
    parser.add_argument('--clean_path', type=str, default='/home/yu34/datasets/UCM_FULL', help='path of clean images (training set)')
    parser.add_argument('--adv_path', type=str, default='/home/yu34/AdvD/UAE-RS/UCM_purified/cw/alexnet/1ksteps_noise_level', help='path of adversarial purified images')
    args = parser.parse_args()
    return args

def collate_fn(batch):
    batched_data=[tfs.ToTensor()(data['image']) for data in batch]
    batched_data=torch.stack(batched_data)
    return batched_data

def fid_score_calc(path_clean, path_adv, model):
    # clean_dataset=ImageFolder(path_clean)
    # adv_dataset=ImageFolder(path_adv)

    clean_dataset = load_dataset("imagefolder", data_dir=path_clean, split='train')
    adv_dataset = load_dataset("imagefolder", data_dir=path_adv, split='train')

    clean_loader=DataLoader(clean_dataset, batch_size=100, collate_fn=collate_fn)
    adv_loader=DataLoader(adv_dataset, batch_size=100, collate_fn=collate_fn)

    fid = piqa.FID()

    clean_features, adv_features=[], []
    for _, images in enumerate(clean_loader):
        images=images.cuda()
        features=model(images)['feat']
        features=nn.AdaptiveAvgPool2d((1,1))(features).squeeze()
        clean_features.append(features.detach().cpu())

    for _, images in enumerate(adv_loader):
        images=images.cuda()
        features=model(images)['feat']
        features=nn.AdaptiveAvgPool2d((1,1))(features).squeeze()
        adv_features.append(features.detach().cpu())

    clean_dist=torch.cat(clean_features)
    adv_dist=torch.cat(adv_features)

    fid_score=fid(clean_dist, adv_dist).item()
    return fid_score

def anls(clean_path, adv_path, model, noise_levels=range(10,130,10)):
    best_fid_score=9999
    optimal_noise_level=0
    for noise_level in noise_levels:
        adv_path_nl=adv_path+"_"+ str(noise_level)
        fid_score=fid_score_calc(path_clean=clean_path,path_adv=adv_path_nl,model=model)
        if fid_score<best_fid_score:
            best_fid_score=fid_score
            optimal_noise_level=noise_level
    return best_fid_score, optimal_noise_level

def get_network(root_path, model, dataset):
    if dataset=='AID':
        num_classes=30
        task="cls"
    elif dataset=='UCM':
        num_classes=21
        task="cls"
    elif dataset=='Vaihingen':
        num_classes=6
        task="seg"
    elif dataset=='Zurich':
        num_classes=9
        task="seg"


    if task=="cls":
        pretrained_path=os.path.join(root_path, dataset, "Pretrain", model)
        file_name=os.listdir(pretrained_path)
        full_file_name=os.path.join(pretrained_path,file_name[0])
        pretrained_state_dict=torch.load(full_file_name)

        new_state_dict = OrderedDict()
        for k, v in pretrained_state_dict.items():
            name = k[7:] # remove `module.`
            new_state_dict[name] = v
        
        pretrained_state_dict=new_state_dict

        if model=='alexnet':
            model = models_cls.alexnet(pretrained=False)
            model.classifier._modules['6'] = nn.Linear(4096, num_classes) 
            model=model.cuda()        
            model.load_state_dict(pretrained_state_dict)
            model=IntermediateLayerGetter(model,{'avgpool':'feat'})   
        elif model=='resnet18':
            model = models_cls.resnet18(pretrained=False)  
            model.fc = torch.nn.Linear(model.fc.in_features, num_classes)        
            model=model.cuda()
            model.load_state_dict(pretrained_state_dict)
            model=IntermediateLayerGetter(model,{'avgpool':'feat'})   
        elif model=='densenet121':
            model = models_cls.densenet121(pretrained=False)
            model.classifier = nn.Linear(1024, num_classes)
            model=model.cuda()
            model.load_state_dict(pretrained_state_dict)
            model=IntermediateLayerGetter(model,{'features':'feat'})   
        elif model=='regnet_x_400mf':
            model = models_cls.regnet_x_400mf(pretrained=False)  
            model.fc = torch.nn.Linear(model.fc.in_features, num_classes)
            model=model.cuda()
            model.load_state_dict(pretrained_state_dict)
            model=IntermediateLayerGetter(model,{'avgpool':'feat'})  

    elif task=="seg":
        pretrained_path=os.path.join(root_path, "segmentation", dataset, "Pretrain", model)
        file_name=os.listdir(pretrained_path)
        full_file_name=os.path.join(pretrained_path,file_name[0])
        pretrained_state_dict=torch.load(full_file_name)

        # new_state_dict = OrderedDict()
        # for k, v in pretrained_state_dict.items():
        #     name = k[7:] # remove `module.`
        #     new_state_dict[name] = v
        
        # pretrained_state_dict=new_state_dict

        if model=='fcn8s':
            model = models_seg.fcn8s(n_classes=num_classes)
            model=model.cuda()        
            model.load_state_dict(pretrained_state_dict)
            model=IntermediateLayerGetter(model,{'conv_block5':'feat'})   
        elif model=='unet':
            model = models_seg.unet(n_classes=num_classes)   
            model=model.cuda()
            model.load_state_dict(pretrained_state_dict)
            model=IntermediateLayerGetter(model,{'down4':'feat'})  
        elif model=='pspnet':
            model = models_seg.pspnet(n_classes=num_classes)
            model=model.cuda()
            model.load_state_dict(pretrained_state_dict)
            model=IntermediateLayerGetter(model,{'res_block5':'feat'})   
        elif model=='linknet':
            model = models_seg.linknet(n_classes=num_classes)  
            model=model.cuda()
            model.load_state_dict(pretrained_state_dict)
            model=IntermediateLayerGetter(model,{'encoder4':'feat'}) 
    model.eval()
    return model

if __name__=="__main__":
    args=args()
    network=get_network(args.root_path,args.model,args.dataset)
    best_fid_score, optimal_noise_level = anls(clean_path=args.clean_path, adv_path=args.adv_path, model=network)
    print(f'The optimal noise level is {optimal_noise_level}, corresponding best_fid_score is {best_fid_score}.')
