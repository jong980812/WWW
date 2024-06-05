import argparse
import pickle as pkl
import numpy as np
import random
import torch
import torchvision as tv


def parse_args():
    parser = argparse.ArgumentParser(description='Say hello')
    parser.add_argument('--map_root', default='.', help='Path to utils')
    parser.add_argument('--concept_root', default='./utils', help='Path to concept')
    parser.add_argument('--interpret_root', default='./utils/heat', help='Path to concept')

    return parser.parse_args()

args = parse_args()


with open('./utils/class_shap.pkl', "rb") as f:
    shap = pkl.load(f)

with open(f'{args.map_root}/heatmap_info/sc_idx.pkl', "rb") as f:
    sc_idx = pkl.load(f)

with open(f'{args.map_root}/heatmap_info/cos.pkl', "rb") as f:
    cos = pkl.load(f)

# with open(f'{args.map_root}/heatmap_info/cos_gt.pkl', "rb") as f: ##
#     cos_gt = pkl.load(f)

with open(f'{args.concept_root}/fc/concept_80k/www_80k_tem_adp_95.pkl', "rb") as f:
    www_major_fc, _= pkl.load(f)

with open(f'{args.concept_root}/l4/concept_80k/www_80k_tem_adp_95.pkl', "rb") as f:
    www_major_l4, _ = pkl.load(f)

# with open(f'{args.concept_root}/www_img_val_80k_tem_adp_10_layer4_minor.pkl', "rb") as f:
#     www_minor_l4, _ = pkl.load(f)

transform = tv.transforms.Compose([
            tv.transforms.Resize(256),
            tv.transforms.CenterCrop(224),
            tv.transforms.ToTensor(),
            tv.transforms.Normalize(mean=[0.98, 0.98, 0.98],
                                    std=[0.065, 0.065, 0.065]),
        ])

examples_fc = tv.datasets.ImageFolder('./images/example_val_fc', transform=transform)
examples_l4 = tv.datasets.ImageFolder('./images/example_val_l4', transform=transform)
example_loader_fc = torch.utils.data.DataLoader(examples_fc, batch_size=1, shuffle=False)
example_loader_l4 = torch.utils.data.DataLoader(examples_l4, batch_size=1, shuffle=False)
interpret_idx = random.randint(0, len(sc_idx)-1) # [0, 2799] 범위의 정수 난수 생성
print('Neurons to interpret: ', interpret_idx)

WWW_concepts = []
WWW_L4_major_concepts = []
WWW_L4_minor_concepts = []
GT = []
GT_idx = []
l4_idx=[]

all_words = []

with open('./utils/imagenet_labels.txt', 'r') as f:  # directory of imagenet_labels.txt
    words = (f.read()).split('\n')

for i in range(len(words)):
    temp=[]
    temp_words = words[i].split(', ')
    for word in temp_words:
        temp.append(f'{word}')
    all_words.append(temp)

correct = False

# for i, major_idx in enumerate(interpret_idx):
for i, major_idx in enumerate(sc_idx[interpret_idx]):
    for gt_concept in all_words[major_idx]: # len(all_words) = 1000 (out of range 에러 : sc_idx 내에 들어갈 수 있는 제일 큰 값이 2047임)
        if gt_concept in www_major_fc[major_idx]:
            correct = True
            break
    if correct:
        print(f'Neuron {i}: {major_idx}')
        print(f'Ground truth: {all_words[major_idx]}')
        print(f'WWW-fc major concept: {www_major_fc[major_idx]}')
        WWW_concepts.append(www_major_fc[major_idx])
        GT.append(all_words[major_idx])
        GT_idx.append(major_idx)
        
        WWW_temp_l4 = []
        WWW_temp_minor_l4 = []

        l4_important_idx = np.argsort(shap[major_idx], axis=0)[-3:]
        l4_idx.append(l4_important_idx)
        for j, minor_idx in enumerate(l4_important_idx):
            print(f'WWW-l4 major {j}: {www_major_l4[minor_idx]}')
            # print(f'WWW-l4 minor {j}: {www_minor_l4[minor_idx]}')
            WWW_temp_l4.append(www_major_l4[minor_idx])
            # WWW_temp_minor_l4.append(www_minor_l4[minor_idx])
        WWW_L4_major_concepts.append(WWW_temp_l4)
        # WWW_L4_minor_concepts.append(WWW_temp_minor_l4)
    correct = False

concepts = []
concepts_l4 = []
for i in range(len(WWW_concepts)):
    concepts.append([GT_idx[i], WWW_concepts[i], GT[i]])
    concepts_l4.append([l4_idx[i], WWW_L4_major_concepts[i]])
    # concepts_l4.append([l4_idx[i], WWW_L4_major_concepts[i], WWW_L4_minor_concepts[i]])

with open(f'{args.interpret_root}/concept_FC.pkl', "wb") as f:
    pkl.dump(concepts, f)

with open(f'{args.interpret_root}/concept_L4.pkl', "wb") as f:
    pkl.dump(concepts_l4, f)
