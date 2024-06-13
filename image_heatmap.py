import argparse
import torch
import numpy as np
import torch.nn as nn
import torchvision as tv
import cv2
import os
from tqdm import tqdm
from matplotlib.colors import ListedColormap
import matplotlib.pyplot as plt
import matplotlib
import colorsys
import pickle as pkl
from torch.nn.functional import cosine_similarity, softmax


def parse_args():
    parser = argparse.ArgumentParser(description='Say hello')
    parser.add_argument('--example_root', default='./datasets/ILSVRC-2012/val', help='Path to D_probe')
    parser.add_argument('--heatmap_save_root', default='./heatmap', help='Path to saved img')
    parser.add_argument('--num_example', default=50, type=int, help='# of examples to be used')
    parser.add_argument('--shapley_root', default='./utils', help='Path to utils')
    parser.add_argument('--map_root', default='./heatmap_info', help='Path to utils')
    parser.add_argument('--model', default='resnet50',type=str, help='Path to utils')

    return parser.parse_args()

def show(img, **kwargs):
    img = np.array(img)
    if img.shape[0] == 3:
        img = img.transpose(1, 2, 0)

    img -= img.min();img /= img.max()
    plt.imshow(img, **kwargs); plt.axis('off')

def get_alpha_cmap(cmap):
  if isinstance(cmap, str):
    cmap = plt.get_cmap(cmap)
  else:
    c = np.array((cmap[0]/255.0, cmap[1]/255.0, cmap[2]/255.0))

    cmax = colorsys.rgb_to_hls(*c)
    cmax = np.array(cmax)
    cmax[-1] = 1.0

    cmax = np.clip(np.array(colorsys.hls_to_rgb(*cmax)), 0, 1)
    cmap = matplotlib.colors.LinearSegmentedColormap.from_list("", [c,cmax])

  alpha_cmap = cmap(np.arange(256))
  alpha_cmap[:,-1] = np.linspace(0, 0.85, 256)
  alpha_cmap = ListedColormap(alpha_cmap)

  return alpha_cmap

def concept_attribution_maps(cmaps, args, model, example_loader, num_top_neuron=5, percentile=90, alpha=0.7, gt=False):
    
    # with open(f"{args.util_root}/heat/class_shap.pkl", "rb") as f:
    with open(args.shapley_root, "rb") as f:
        shap_value = pkl.load(f) # 2 * 2048

    c_heatmap = []
    s_heatmap = []
    cc_val = []
    sc_val = []
    sc_idx = []

    #### Class concept Atribute Maps #### heatmap 내의 폴더명(0000,0001)은 실제 레이블 
    print("Starting Class concept Atribute Maps...")
    for j, (img, label) in enumerate(example_loader):
        os.makedirs(f'{args.heatmap_save_root}/{label.item():04d}/class_attribute_n{num_top_neuron}_p{percentile}_a90', exist_ok=True)
        show(img[0])        
        img = img.cuda()    
        feature_maps = model.extract_feature_map_4(img) # 1*3*224*224 => 1*2048*7*7
        predict = model(img) # 1 * 2 => model을 우리가 사용하는 데이터 모델 사용해야됨(1000인건 imagenet으로 한 결과고, 클래스 수만큼의 텐서 크기를 가짐)
        predict = predict[0].cpu().detach().numpy()
        predict = np.argmax(predict) # 0 or 1이 나옴
        feature_maps = feature_maps[0].cpu().detach().numpy()
        feature_maps = feature_maps.transpose(1, 2, 0) # 7*7*2048
        if gt:
            most_important_concepts = np.argsort(shap_value[label.item()])[::-1][:num_top_neuron]
        else:
            most_important_concepts = np.argsort(shap_value[predict])[::-1][:num_top_neuron] # predict에 해당하는 클래스의 shap_value 내림차순 정렬하고 num_top_neuron 만큼 뽑아 인덱스 반환
        
        for i, c_id in enumerate(most_important_concepts):
            cmap = cmaps[i] # blue, red, yellow, green, purple 순으로 중요한 concept(shap_value 값 가장 큰 인덱스)
            heatmap = feature_maps[:, :, c_id] # 7*7 (2048 중에 c_id 번째 feature_map 꺼내옴)

            sigma = np.percentile(feature_maps[:,:,c_id].flatten(), percentile) # 백분위 90% (상위 10%의 값)
            heatmap = heatmap * np.array(heatmap > sigma, np.float32)

            heatmap = cv2.resize(heatmap[:, :, None], (224, 224)) # 7*7 => 224*224
            show(heatmap, cmap=cmap, alpha=0.9) # 원본이미지 위에 히트맵을 덮는 코드
        # plt.show()
        
        if gt:
            plt.savefig(f'{args.heatmap_save_root}/{label.item():04d}/class_attribute_n{num_top_neuron}_p{percentile}_a90/Class_att_gt{label.item():04d}_{(j%args.num_example):02d}.jpg')
        else:
            # .jpg 파일명 : 실제 레이블 이름의 폴더안에 predict된 레이블 이름.jpg 형태임
            plt.savefig(f'{args.heatmap_save_root}/{label.item():04d}/class_attribute_n{num_top_neuron}_p{percentile}_a90/Class_att_{predict:04d}_{(j%args.num_example):02d}.jpg')
        plt.clf()

        if j % 20 == 0:
            print(j)


    #### Class overall Atribute Maps ####
    print("\nStarting Class overall Atribute Maps...")
    for j, (img, label) in enumerate(example_loader):
        os.makedirs(f'{args.heatmap_save_root}/{label.item():04d}/class_overall_n{num_top_neuron}_p0_a50', exist_ok=True)
        show(img[0]) 
        img = img.cuda()    
        feature_maps = model.extract_feature_map_4(img)
        feature_maps = feature_maps[0].cpu().detach().numpy()
        feature_maps = feature_maps.transpose(1, 2, 0)
        predict = model(img)
        predict = predict[0].cpu().detach().numpy()
        predict = np.argmax(predict)
        if gt:
            most_important_concepts = np.argsort(shap_value[label.item()])[::-1][:num_top_neuron]
        else:
            most_important_concepts = np.argsort(shap_value[predict])[::-1][:num_top_neuron]
        overall_heatmap = np.zeros((224, 224))
        temp_weight = []
        
        for i, c_id in enumerate(most_important_concepts):
            cmap = cmaps[i]
            heatmap = feature_maps[:, :, c_id]

            # sigma = np.percentile(feature_maps[:,:,c_id].flatten(), percentile)
            # heatmap = heatmap * np.array(heatmap > sigma, np.float32)

            heatmap = cv2.resize(heatmap[:, :, None], (224, 224))
            if gt:
                weight = shap_value[label.item()][c_id] / np.sum(shap_value[label.item()][most_important_concepts])
            else:
                weight = shap_value[predict][c_id] / np.sum(shap_value[predict][most_important_concepts])
            overall_heatmap += heatmap * weight
            temp_weight.append(weight)
        
        c_heatmap.append(overall_heatmap)
        cc_val.append(temp_weight)
        show(overall_heatmap, cmap='Reds', alpha=0.5)

        
        if gt:
            plt.savefig(f'{args.heatmap_save_root}/{label.item():04d}/class_overall_n{num_top_neuron}_p0_a50/Class_ovr_gt{label.item():04d}_{(j%args.num_example):02d}.jpg')
        else:
            # .jpg 파일명 : 실제 레이블 이름의 폴더안에 predict된 레이블 이름.jpg 형태임
            plt.savefig(f'{args.heatmap_save_root}/{label.item():04d}/class_overall_n{num_top_neuron}_p0_a50/Class_ovr_{predict:04d}_{(j%args.num_example):02d}.jpg')
        plt.clf()
        # plt.close()
        if j % 20 == 0:
            print(j)
    
    with open(f"{args.map_root}/cc_val.pkl", "wb") as f:
        pkl.dump(cc_val, f)
    cc_val = None

    with open(f"{args.map_root}/c_heatmap.pkl", "wb") as f:
        pkl.dump(c_heatmap, f)
    c_heatmap = None

    #### sample concept Atribute Maps ####
    print("\nStarting sample concept Atribute Maps...")
    for j, (img, label) in enumerate(example_loader): # len(example_loader) = 2800 (그래서 sc_idx 읽어보면 len():2800임)
        os.makedirs(f'{args.heatmap_save_root}/{label.item():04d}/sample_attribute_n{num_top_neuron}_p{percentile}_a90', exist_ok=True)
        show(img[0])
        img = img.cuda() # 1*3*224*224
        feature_maps = model.extract_feature_map_4(img) # 1*2048*7*7
        predict = model(img) 
        predict = predict[0].cpu().detach().numpy() #len(predict) = 2
        predict = np.argmax(predict) # 위에서 predict가 확률값으로 나오는데 둘중 큰값의 인덱스 반환
        sample_shap = model._compute_taylor_scores(img, predict) # len(sample_shap) = 2 (first_order_taylor_scores, output(len():2) 에 대한 텐서 2개)
        sample_shap = sample_shap[0][0][0,:,0,0] # len() : 2048
        sample_shap = sample_shap.cpu().detach().numpy()
        feature_maps = feature_maps[0].cpu().detach().numpy() # 2048*7*7
        feature_maps = feature_maps.transpose(1, 2, 0) #7*7*2048
        most_important_concepts = np.argsort(sample_shap)[::-1][:num_top_neuron] # 2048개 중에 상위 num_top_neuron(default=3)개의 중요 컨셉 뽑음
        sc_idx.append(most_important_concepts)
        
        for i, c_id in enumerate(most_important_concepts):
            cmap = cmaps[i]
            heatmap = feature_maps[:, :, c_id]

            sigma = np.percentile(feature_maps[:,:,c_id].flatten(), percentile)
            heatmap = heatmap * np.array(heatmap > sigma, np.float32)

            heatmap = cv2.resize(heatmap[:, :, None], (224, 224))
            show(heatmap, cmap=cmap, alpha=0.9)
        
        plt.savefig(f'{args.heatmap_save_root}/{label.item():04d}/sample_attribute_n{num_top_neuron}_p{percentile}_a90/sample_att_{predict:04d}_{(j%args.num_example):02d}.jpg')
        plt.clf()
        if j % 20 == 0:
            print(j)

    with open(f"{args.map_root}/sc_idx.pkl", "wb") as f:
        pkl.dump(sc_idx, f)
    sc_idx = None

    #### Sample overall Atribute Maps ####
    print("sample overall Atribute Maps")
    for j, (img, label) in enumerate(example_loader):
        os.makedirs(f'{args.heatmap_save_root}/{label.item():04d}/sample_overall_n{num_top_neuron}_p0_a50', exist_ok=True)
        show(img[0])
        img = img.cuda()    
        feature_maps = model.extract_feature_map_4(img)
        predict = model(img)
        predict = predict[0].cpu().detach().numpy()
        predict = np.argmax(predict)
        sample_shap = model._compute_taylor_scores(img, predict)
        sample_shap = sample_shap[0][0][0,:,0,0]
        sample_shap = sample_shap.cpu().detach().numpy()
        feature_maps = feature_maps[0].cpu().detach().numpy()
        feature_maps = feature_maps.transpose(1, 2, 0)
        most_important_concepts = np.argsort(sample_shap)[::-1][:num_top_neuron]
        overall_heatmap = np.zeros((224, 224))
        temp_weight = []
        for i, c_id in enumerate(most_important_concepts):
            cmap = cmaps[i]
            heatmap = feature_maps[:, :, c_id]

            # sigma = np.percentile(feature_maps[:,:,c_id].flatten(), percentile)
            # heatmap = heatmap * np.array(heatmap > sigma, np.float32)

            heatmap = cv2.resize(heatmap[:, :, None], (224, 224))
            weight = sample_shap[c_id] / np.sum(sample_shap[most_important_concepts])
            overall_heatmap += heatmap * weight
            temp_weight.append(weight)

        s_heatmap.append(overall_heatmap)
        sc_val.append(temp_weight)
        show(overall_heatmap, cmap='Reds', alpha=0.5)

        plt.savefig(f'{args.heatmap_save_root}/{label.item():04d}/sample_overall_n{num_top_neuron}_p0_a50/sample_ovr_{predict:04d}_{(j%args.num_example):02d}.jpg')
        plt.clf()
        if j % 20 == 0:
            print(j)

    with open(f"{args.map_root}/sc_val.pkl", "wb") as f:
        pkl.dump(sc_val, f)
    sc_val = None
    with open(f"{args.map_root}/s_heatmap.pkl", "wb") as f:
        pkl.dump(s_heatmap, f)  
    s_heatmap = None

def compute_heatmap_cosine_similarity(args):

    with open(args.map_root + "/c_heatmap.pkl", "rb") as f:
        c_heatmap = pkl.load(f)

    with open(args.map_root + "/s_heatmap.pkl", "rb") as f:
        s_heatmap = pkl.load(f)

    cos = []

    for i in range(len(c_heatmap)):
        c_heatmap[i] = torch.from_numpy(c_heatmap[i].flatten())
        s_heatmap[i] = torch.from_numpy(s_heatmap[i].flatten())
        cos.append(cosine_similarity(c_heatmap[i].reshape(1, -1), s_heatmap[i].reshape(1, -1)).item())

    with open(args.map_root + "/cos.pkl", "wb") as f:
        pkl.dump(cos, f)

def main():
    
    args = parse_args()
    if args.model =='resnet50':
        
    ## Load model ##
    ##### ResNET50 #####
        from models.resnet import resnet50, ResNet50_Weights
        weights = ResNet50_Weights.DEFAULT
        model = resnet50(weights=weights)    

        weights_path ='/data/psh68380/repos/WWW/checkpoint-2.pth'
        model.fc = torch.nn.Linear(2048, 2)#! head를 pth랑 맞춰줌
        
        pretrained_weights = torch.load(weights_path,map_location='cpu')#! pth를 읽어서 변수에 담음. 
        #! pretrained_weights['model']-> 이게 weight고 디버거에서 찍어보고
        print(f"Load pretrained from {weights_path}")
        print(model.load_state_dict(pretrained_weights['model'],strict=True))
        #!print()-> all key matching
        for name, layer in model.named_modules():
            if isinstance(layer, torch.nn.Conv2d):
                layer.padding_mode = 'replicate'
    elif args.model =='vit':
        from models.ViT import _create_vision_transformer
        model_kwargs = dict(patch_size=16, embed_dim=768, depth=12, num_heads=12)
        model = _create_vision_transformer('vit_base_patch16_224', pretrained=True, **dict(model_kwargs))
    model = model.cuda()
    model.eval()
    featdim = 2048

    transform = tv.transforms.Compose([
            tv.transforms.Resize(256),
            tv.transforms.CenterCrop(224),
            tv.transforms.ToTensor(),
            tv.transforms.Normalize(mean=[0.98, 0.98, 0.98],
                                    std=[0.065, 0.065, 0.065]),
        ])
    
    os.makedirs(f'{args.map_root}', exist_ok=True)

    examples = tv.datasets.ImageFolder(args.example_root, transform=transform)
    example_loader = torch.utils.data.DataLoader(examples, batch_size=1, shuffle=False) # 여기서 나온게 2800개

    cmaps = [
        get_alpha_cmap((54, 197, 240)),##blue
        get_alpha_cmap((210, 40, 95)),##red
        get_alpha_cmap((236, 178, 46)),##yellow
        get_alpha_cmap((15, 157, 88)),##green
        get_alpha_cmap((84, 25, 85)),##purple
        get_alpha_cmap((255, 0, 0))##real red
    ]

    concept_attribution_maps(cmaps, args, model, example_loader, num_top_neuron=3, percentile=70, alpha=0.8, gt=False)
    compute_heatmap_cosine_similarity(args)

if __name__ == '__main__':
    main()


