import argparse
import torchvision as tv
import torch
import numpy as np
from tqdm import tqdm
import os
import pickle
import timm
from models import ViT
def parse_args():
    parser = argparse.ArgumentParser(description='Say hello')
    parser.add_argument('--data_root', default='./datasets/ILSVRC-2012/train', help='Path to ImageNet data')
    parser.add_argument('--shap_save_root', default='./utils/class_shap.pkl', help='Path to Shapley value matrix data')
    parser.add_argument('--model',default='resnet50',)
    parser.add_argument('--target_layer',default='head')
    return parser.parse_args()


def main():
    
    args = parse_args()

    ## Load model ##
    ##### ResNET50 #####
    if args.model=='resnet50':
        from models.resnet import resnet50, ResNet50_Weights
    
        weights = ResNet50_Weights.DEFAULT
        model = resnet50(weights=weights)
        model.fc = torch.nn.Linear(2048, 2)#! head를 pth랑 맞춰줌
        # weights_path ='/data/psh68380/repos/WWW/checkpoint-2.pth'
        # pretrained_weights = torch.load(weights_path,map_location='cpu') #! pth를 읽어서 변수에 담음. 
        # #! pretrained_weights['model']-> 이게 weight고 디버거에서 찍어보고
        # print(f"Load pretrained from {weights_path}")
        # print(model.load_state_dict(pretrained_weights['model'],strict=True))
        # #!print()-> all key matching
        # for name, layer in model.named_modules():
        #     if isinstance(layer, torch.nn.Conv2d):
        #         layer.padding_mode = 'replicate'
        featdim = 2 if args.target_layer =='head' else 2048
        class_dim = 2
    elif args.model =='vit':
        model = timm.create_model('vit_tiny_patch16_224',True)
        featdim = 1000
        class_dim = 1000
    '''
    model.fc. 어쩌구 해서 shape바꿈
    
    '''
    # weights_path ='/data/psh68380/repos/WWW/resnet_mw.pth'
    model = model.cuda()
    model.eval()


    transform = tv.transforms.Compose([
            tv.transforms.Resize(256),
            tv.transforms.CenterCrop(224),
            tv.transforms.ToTensor(),
            tv.transforms.Normalize(mean=[0.98, 0.98, 0.98],
                                    std=[0.065, 0.065, 0.065]),
        ])

    traindata = tv.datasets.ImageFolder(args.data_root, transform=transform)
    trainloader = torch.utils.data.DataLoader(traindata, batch_size=1, shuffle=False, pin_memory=True, num_workers=8)
    if args.model=='resnet50':
        if not os.path.exists(args.shap_save_root):
            class_num = 0
            shap = []
            shap_class = np.zeros((class_dim, featdim)) 
            shap_temp = np.zeros(featdim)
            c = 0
            for image, labels in (trainloader):
                image = image.cuda()
                c+=1
                if c%100==0:print(c) 
                if class_num != labels:
                    for i in range(len(shap)):
                        shap_temp += shap[i].squeeze()    
                    shap_class[class_num,:] = shap_temp / len(shap)
                    shap = []
                    shap_temp = np.zeros(featdim)
                    class_num += 1
                        
                shap_batch  = model._compute_taylor_scores(image, labels,args.target_layer)
                shap.append(shap_batch[0][0].squeeze().cpu().detach().numpy())

            for i in range(len(shap)):
                shap_temp += shap[i].squeeze()    
            shap_class[class_num,:] = shap_temp / len(shap)
            shap = []

            with open(args.shap_save_root, 'wb') as f:
                pickle.dump(shap_class, f)
    elif args.model=='vit':
        if not os.path.exists(args.shap_save_root):
            class_num = 0
            shap = []
            shap_class = np.zeros((class_dim, featdim)) 
            shap_temp = np.zeros(featdim)
            c = 0
            for image, labels in (trainloader):
                image = image.cuda()
                c+=1
                if c%100==0:print(c) 
                if class_num != labels:
                    for i in range(len(shap)):
                        shap_temp += shap[i].squeeze()    
                    shap_class[class_num,:] = shap_temp / len(shap)
                    shap = []
                    shap_temp = np.zeros(featdim)
                    class_num += 1
                        
                shap_batch  = model._compute_taylor_scores(image, labels)
                shap.append(shap_batch[0][0].squeeze().cpu().detach().numpy())

            for i in range(len(shap)):
                shap_temp += shap[i].squeeze()    
            shap_class[class_num,:] = shap_temp / len(shap)
            shap = []

            with open(args.shap_save_root, 'wb') as f:
                pickle.dump(shap_class, f)
if __name__ == '__main__':
    main()