import argparse
import clip
import torch
import os
import torchvision as tv
import nltk
from nltk.corpus import wordnet as wn
import numpy as np
import pickle
from tqdm import tqdm
from torch.nn.functional import cosine_similarity

def parse_args():
    parser = argparse.ArgumentParser(description='Say hello')
    parser.add_argument('--word_save_root', default='./utils/words_only.pkl', help='Path to word features')
    parser.add_argument('--img_save_root', default='./images/example_val_final', help='Path to saved img')
    parser.add_argument('--img_feat_root', default='./utils', help='Path to img features')
    parser.add_argument('--concept_sim_root', default='./utils/', help='Path to concept idx data') 
    parser.add_argument('--concept_sim_num', default=4, type=int, help='# of split concept sim')
    parser.add_argument('--concept_root', default='./utils', help='Path to concept')
    parser.add_argument('--num_example', default=40, type=int, help='# of examples to be used')
    parser.add_argument('--alpha', default=95, type=int, help='# of concept to select in img')
    parser.add_argument('--layer', default='fc', help='target layer')

    return parser.parse_args()

def text_to_feature(all_words, model, device, args, template=False):
    word_features = []
    # for word in tqdm(all_words):
    i=0
    for word in all_words:
        text_inputs = clip.tokenize(word).to(device)
        with torch.no_grad():
            text_features = model.encode_text(text_inputs)
        word_features.append(text_features.cpu().numpy())

        i += 1
        if i % 100 == 0:
            print(i)
    print("text_to_feature finish.")

    if not template:
        word_features = np.concatenate(word_features, axis=0)
        with open(args.word_save_root, 'wb') as f:
            pickle.dump(word_features, f)
    else:
        word_features = np.array(word_features)
    return word_features

def load_word_features(args):
    with open(args.word_save_root, 'rb') as f:
        word_features = pickle.load(f)
    return word_features

def img_to_features(model, device, preprocess, args, detail=False):
    if detail:
        img_dir = args.crop_save_root
    else:
        img_dir = args.img_save_root
    imgset = tv.datasets.ImageFolder(img_dir)
    img_features = []

    with torch.no_grad():
        # for image, labels in tqdm(imgset):
        i=0
        for image, labels in imgset:
            image = preprocess(image).unsqueeze(0).to(device)
            image_feature = model.encode_image(image)
            image_feature = image_feature.cpu().numpy()
            img_features.append(image_feature)

            i+=1
            if i % 1000 == 0:
                print("img_to_features : %d / %d" %(i, len(imgset)))

        img_features = np.concatenate(img_features, axis=0)
        if detail:
            img_feature_root = f'{args.img_feat_root}/crop_features_{args.num_example}.pkl'
        else:
            img_feature_root = f'{args.img_feat_root}/img_features_{args.num_example}.pkl'
        with open(img_feature_root, 'wb') as f:
            pickle.dump(img_features, f)
        
        print("\nimg_to_features finish.")
    return img_features

def load_img_features(args, detail=False):
    if detail:
        with open(f'{args.img_feat_root}/crop_features_{args.num_example}.pkl', 'rb') as f:
            img_features = pickle.load(f)
    else:
        with open(f'{args.img_feat_root}/img_features_{args.num_example}.pkl', 'rb') as f:
            img_features = pickle.load(f)
    return img_features

def compute_concept_similarity(img_features, word_features, args, template_features=None, device='cuda', detail=False, adaptive=False):
    concept_sim = []
    counter = 0
    img_features = torch.Tensor(img_features).to(device)
    word_features = torch.Tensor(word_features).to(device)
    if adaptive:
        template_features = torch.Tensor(template_features).to(device)

    if adaptive:
        # for i in tqdm(range(len(img_features))):
        n=0
        for i in range(len(img_features)):
            n+=1
            if n % 1000 == 0:
                print("compute_concept_similarity : %d / %d" %(i, len(img_features)))


            if i % (len(img_features)//args.concept_sim_num) == 0 and i != 0:
                concept_sim = np.concatenate(concept_sim, axis=0)
                if detail:
                    with open(args.concept_sim_root +'/crop_adaptive_sim_'+ str(args.num_example)+'_'+str(counter)+'.pkl', 'wb') as f:
                        pickle.dump(concept_sim, f)
                else:
                    with open(args.concept_sim_root +'/concept_adaptive_sim_'+ str(args.num_example)+'_'+str(counter)+'.pkl', 'wb') as f:
                        pickle.dump(concept_sim, f)
                concept_sim = []
                counter += 1
            img = img_features[i].reshape(1, -1)
            sim = cosine_similarity(img, word_features)
            template_sim = cosine_similarity(img, template_features)
            sim = sim - template_sim
            concept_sim.append([sim.cpu().numpy()])

        concept_sim = np.concatenate(concept_sim, axis=0)
        if detail:
            with open(args.concept_sim_root +'/crop_adaptive_sim_'+ str(args.num_example)+'_'+str(counter)+'.pkl', 'wb') as f:
                pickle.dump(concept_sim, f)
        else:
            with open(args.concept_sim_root +'/concept_adaptive_sim_'+ str(args.num_example)+'_'+str(counter)+'.pkl', 'wb') as f:
                pickle.dump(concept_sim, f)
        print("compute_concept_similarity finish.")
        
    else:
        # for i in tqdm(range(len(img_features))):
        n=0
        for i in range(len(img_features)):
            n+=1
            if n % 1000 == 0:
                print("compute_concept_similarity : %d / %d" %(i, len(img_features)))


            if i % (len(img_features)//args.concept_sim_num) == 0 and i != 0:
                concept_sim = np.concatenate(concept_sim, axis=0)
                if detail:
                    with open(args.concept_sim_root +'/crop_sim_'+ str(args.num_example)+'_'+str(counter)+'.pkl', 'wb') as f:
                        pickle.dump(concept_sim, f)
                else:
                    with open(args.concept_sim_root +'/concept_sim_'+ str(args.num_example)+'_'+str(counter)+'.pkl', 'wb') as f:
                        pickle.dump(concept_sim, f)
                concept_sim = []
                counter += 1
            img = img_features[i].reshape(1, -1)
            sim = cosine_similarity(img, word_features)
            concept_sim.append([sim.cpu().numpy()])

        concept_sim = np.concatenate(concept_sim, axis=0)
        if detail:
            with open(args.concept_sim_root +'/crop_sim_'+ str(args.num_example)+'_'+str(counter)+'.pkl', 'wb') as f:
                pickle.dump(concept_sim, f)
        else:
            with open(args.concept_sim_root +'/concept_sim_'+ str(args.num_example)+'_'+str(counter)+'.pkl', 'wb') as f:
                pickle.dump(concept_sim, f)
        print("compute_concept_similarity finish.")

def concept_discovery(all_synsets, args, all_words_wotem=None, detail=False,  adaptive=True, data=1, template=True):
    
    img_weights = []

    for j in range(args.concept_sim_num):
        if detail:
            if adaptive:
                with open(args.concept_sim_root +'/concept_adaptive_sim_'+ str(args.num_example)+'_'+str(j)+'.pkl', 'rb') as f:
                    concept_sim = pickle.load(f)
            else:
                with open(args.concept_sim_root +'/concept_sim_'+str(args.num_example)+'_'+str(j)+'.pkl', 'rb') as f:
                    concept_sim = pickle.load(f)
        elif adaptive: #여기로 들어와서
            with open(args.concept_sim_root +'/concept_adaptive_sim_'+ str(args.num_example)+'_'+str(j)+'.pkl', 'rb') as f: # 여기서 j:0~4 가져옴
                concept_sim = pickle.load(f)
        else:
            with open(args.concept_sim_root +'/concept_sim_'+str(args.num_example)+'_'+str(j)+'.pkl', 'rb') as f:
                concept_sim = pickle.load(f)
    # 결과적으로 concept_sim : 10000 * 82115

    ##### w/o adaptive thresholding #####
        img_concept_weight = np.zeros(len(concept_sim[0])) # 82115(data 1k 일때)
        for i in range(len(concept_sim)): # 10000
            if i != 0 and i % args.num_example == 0: # num_example : high activating image 40개인듯 (근데 왜? 이걸조건에 넣는거?)
                img_weights.append(img_concept_weight)
                img_concept_weight = np.zeros(len(concept_sim[0]))
            img_concept_weight += concept_sim[i]
        img_weights.append(img_concept_weight) # 1000 * 82115
    
    concept_weight = []
    concept = []

    for i in range(len(img_weights)): # len(img_weights) : 1000
        max_sim = np.max(img_weights[i]) # img_weights[i] : 82115
        threshold = max_sim * (args.alpha/100)
        img_concept_idx = np.where(img_weights[i] > threshold)[0] # threshold 넘는 컨셉의 idx들이 담김
        temp_weight = img_weights[i][img_concept_idx]
        concept_idx = np.argsort(img_weights[i][img_concept_idx])[::-1] # weight 큰 순서대로, 인덱스를 내림차순 정렬해서 저장
        concept_weight.append(temp_weight[concept_idx]) # weight 큰 순서대로 concept_weight에 담길듯
        concpet_words = []

        if data == 1 or data == 20 or data == 365:
            for j in concept_idx:
                if template:
                    word = all_words_wotem[img_concept_idx[j]] # all_words_wotem : 1862인데 img_concept_idx[0]가 54072라서 out of 어쩌구 에러떴었음(concept_adaptive_sim 어쩌구 pickle 파일 새로 돌릴때마다 다 삭제해야됨)
                else:
                    word = all_synsets[img_concept_idx[j]]
                concpet_words.append(word)
            concept.append(concpet_words)
        else:
            for j in concept_idx:
                if template:
                    word = all_words_wotem[img_concept_idx[j]]
                else:
                    word = all_synsets[img_concept_idx[j]]
                concpet_words.append(word)
            concept.append(concpet_words)
    
    if template:
        tem_name = 'tem'
    else:
        tem_name = 'wotem'          

    if adaptive:
        adp_name = 'adp'
    else:
        adp_name = 'woadp'

    with open(f'{args.concept_root}/www_{data}k_{tem_name}_{adp_name}_{args.alpha}.pkl', 'wb') as f: # directory of NM_img_val_1k.pkl
        pickle.dump((concept,concept_weight), f)


def main():

    args = parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, preprocess = clip.load("ViT-B/32", device=device)
    # nltk.download('wordnet')

    all_words = []
    all_synsets = []
    base_template = ['A photo of a']
    template = True    
    adaptive = True
    data = 80         # 1k, 20k, 80k, 365, broaden
    layer = 'l4'      # fc, l4, l3, l2, l1

    args.img_save_root = f'./images/example_val_{layer}'

    if template:
        all_words_wotem = []
    else:
        all_words_wotem = None
        adaptive = False

    if data== 1:
        with open('./utils/imagenet_labels.txt', 'r') as f:  # directory of imagenet_labels.txt
            words = (f.read()).split('\n')
        for i in range(len(words)):
            temp_words = words[i].split(', ')

            for word in temp_words:
                if template:
                    all_words.append(f'A photo of a {word}')
                    all_words_wotem.append(f'{word}')    
                else:
                    all_words.append(f'{word}')
    elif data == 80:
        for synset in wn.all_synsets('n'):
            word = synset.lemma_names()[0].replace('_', ' ')
            if template:
                all_words.append(f'A photo of a {word}')
                all_words_wotem.append(f'{word}')
            else:
                all_words.append(f'{word}')
            all_synsets.append(synset)
    elif data == 365:
        with open("./utils/categories_places365.txt", "r") as f:
            places365_classes = f.read().split("\n")

        for i, cls in enumerate(places365_classes):
            word = cls[3:].split(' ')[0]
            word = word.replace('/', '-')
            if template:
                all_words.append(f'A photo of a {word}')    
                all_words_wotem.append(f'{word}')
            else:
                all_words.append(f'{word}')
        with open("./utils/places365_label.pkl", "wb") as f:
            pickle.dump(all_words_wotem, f)
    elif data == 'broaden':
        with open('./utils/broden_labels_clean.txt', 'r') as f:
            words = (f.read()).split('\n')
        for word in words:
            if template:
                all_words.append(f'A photo of a {word}')    
                all_words_wotem.append(f'{word}')
            else:
                all_words.append(f'{word}')

    if not os.path.exists(args.word_save_root):
        word_features = text_to_feature(all_words, model, device, args)
        template_features = text_to_feature(base_template, model, device, args, template=template)
    else:
        word_features = load_word_features(args)
        template_features = text_to_feature(base_template, model, device, args, template=template)

    if not os.path.exists(f'{args.img_feat_root}/img_features_{args.num_example}.pkl'):
        img_features = img_to_features(model, device, preprocess, args)
    else:
        img_features = load_img_features(args)

    if not os.path.exists(args.concept_sim_root +'/concept_sim_'+str(args.num_example)+'_'+str(args.concept_sim_num-1)+'.pkl'):
        compute_concept_similarity(img_features, word_features, args, template_features=template_features.squeeze(), adaptive=adaptive)

    if True:   
        concept_discovery(all_words, args, all_words_wotem=all_words_wotem, adaptive=adaptive, data=data, template=template)


if __name__ == '__main__':
    main()


