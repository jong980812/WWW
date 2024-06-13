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
    parser.add_argument('--word_save_root', default='./utils/words_only_80k.pkl', help='Path to word features')
    parser.add_argument('--img_save_root', default='.', help='Path to saved img')
    parser.add_argument('--img_feat_root', default='./utils', help='Path to img features')
    parser.add_argument('--concept_sim_root', default='./utils/', help='Path to concept idx data') 
    parser.add_argument('--concept_sim_num', default=4, type=int, help='# of split concept sim(몇개의 파일에 쪼개서 저장할지)')
    parser.add_argument('--concept_root', default='./utils', help='Path to concept')
    parser.add_argument('--num_example', default=40, type=int, help='# of examples to be used')
    parser.add_argument('--alpha', default=95, type=int, help='# of concept to select in img')
    parser.add_argument('--layer', default='fc', help='target layer')
    parser.add_argument('--data_size', default=1,type=str, help='# of concept (1k, 20k, 80k, 365, broaden)')
    parser.add_argument('--detail', default=False, help='True(minor concept), False(major concept)')

    return parser.parse_args()

def text_to_feature(all_words, model, device, args, template=False):
    word_features = []
    # for word in tqdm(all_words):
    i=0
    for word in all_words:
        text_inputs = clip.tokenize(word).to(device)
        with torch.no_grad():
            text_features = model.encode_text(text_inputs) # [1,512]
        word_features.append(text_features.cpu().numpy())

        i += 1
        if i % 100 == 0:
            print(i)

    if not template:
        word_features = np.concatenate(word_features, axis=0) # [71, 512]
        with open(args.word_save_root, 'wb') as f:
            pickle.dump(word_features, f)
    else:
        word_features = np.array(word_features)
    print("text_to_feature finish.")
    
    return word_features

def load_word_features(args):
    with open(args.word_save_root, 'rb') as f:
        word_features = pickle.load(f)
    return word_features

def img_to_features(model, device, preprocess, args, detail=False):
    if detail:
        img_dir = args.img_save_root # crop image (minor)
    else:
        img_dir = args.img_save_root
    imgset = tv.datasets.ImageFolder(img_dir) # (fc) data_point 수 : 80개(레이블당 40개씩) ./images 폴더 안에 있는거
    img_features = []

    with torch.no_grad():
        # for image, labels in tqdm(imgset):
        i=0 
        for image, labels in imgset: # fc layer의 경우 이걸 80번 반복(80개의 data point 있으니까)
            image = preprocess(image).unsqueeze(0).to(device) # 1*3*224*224
            image_feature = model.encode_image(image) 
            image_feature = image_feature.cpu().numpy() # 1*512
            img_features.append(image_feature)

            i+=1
            if i % 1000 == 0:
                print("img_to_features : %d / %d" %(i, len(imgset)))
        # len(img_features) = 20
        img_features = np.concatenate(img_features, axis=0) # 80(images 폴더내의 이미지수) * 512, 20 * 512
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
    img_features = torch.Tensor(img_features).to(device) # 80*512
    word_features = torch.Tensor(word_features).to(device) # 82115*512 (80k), 71*512
    if adaptive: # template_features 1차원
        template_features = torch.Tensor(template_features).to(device) # template_features.shape : [512]

    if img_features.shape[0] // args.num_example < args.concept_sim_num: # 40000 // 40
        args.concept_sim_num = img_features.shape[0] // args.num_example

    # 여기서 4개의 파일이 만들어짐 
    if adaptive:
        # for i in tqdm(range(len(img_features))):
        n=0
        for i in range(len(img_features)): # 40000 ( # of class * 40)
            n+=1
            if n % 1000 == 0:
                print("compute_concept_similarity : %d / %d" %(i, len(img_features)))


            if i % (len(img_features)//args.concept_sim_num) == 0 and i != 0: #concept_sim_num 디폴트가 4임 (fc일땐 20번에 한번씩(20,40,60) if문 명령 실행)
                concept_sim = np.concatenate(concept_sim, axis=0) # 20*82115
                if detail:
                    with open(args.concept_sim_root +'/crop_adaptive_sim_'+ str(args.num_example)+'_'+str(counter)+'.pkl', 'wb') as f:
                        pickle.dump(concept_sim, f)
                else:
                    with open(args.concept_sim_root +'/concept_adaptive_sim_'+ str(args.num_example)+'_'+str(counter)+'.pkl', 'wb') as f:
                        pickle.dump(concept_sim, f)
                concept_sim = []
                counter += 1
            img = img_features[i].reshape(1, -1) # 1*512 , img_features[i] : 512
            sim = cosine_similarity(img, word_features) # 82115
            template_sim = cosine_similarity(img, template_features) # 1
            sim = sim - template_sim # 82115
            concept_sim.append([sim.cpu().numpy()])

        concept_sim = np.concatenate(concept_sim, axis=0) # 20*82115 (61~80번째 샘플에 대한 sim)
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

    for j in range(args.concept_sim_num): # 디폴트 4
        if detail:
            if adaptive:
                with open(args.concept_sim_root +'/crop_adaptive_sim_'+ str(args.num_example)+'_'+str(j)+'.pkl', 'rb') as f:
                    concept_sim = pickle.load(f)
            else:
                with open(args.concept_sim_root +'/crop_sim_'+str(args.num_example)+'_'+str(j)+'.pkl', 'rb') as f:
                    concept_sim = pickle.load(f)
        else:
            if adaptive:
                with open(args.concept_sim_root +'/concept_adaptive_sim_'+ str(args.num_example)+'_'+str(j)+'.pkl', 'rb') as f:
                    concept_sim = pickle.load(f)
            else:
                with open(args.concept_sim_root +'/concept_sim_'+str(args.num_example)+'_'+str(j)+'.pkl', 'rb') as f:
                    concept_sim = pickle.load(f)
    # concept_sim : 20 * 82115 (한 파일 읽을때마다)

    ##### w/o adaptive thresholding #####
        img_concept_weight = np.zeros(len(concept_sim[0])) # 82115(data 80k 일때)
        for i in range(len(concept_sim)): # 20
            if i != 0 and i % args.num_example == 0: # num_example : high activating image 40개인듯 (근데 왜? 이걸조건에 넣는거?)
                img_weights.append(img_concept_weight)
                img_concept_weight = np.zeros(len(concept_sim[0]))
            img_concept_weight += concept_sim[i] # 한 concept_sim 파일마다 20개들어가잖아 그걸 다 합함(82115*1) 
        img_weights.append(img_concept_weight) # 4 * 82115 (파일 4개니까 파일마다 합한걸 한 행씩 집어넣음)
    
    concept_weight = []
    concept = []

    for i in range(len(img_weights)): # len(img_weights) : 4
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
    with open(f'{args.concept_root}/www_{data}k_{tem_name}_{adp_name}_{args.alpha}.pkl','rb') as f:
        data = pickle.load(f)
    with open(f'{args.concept_root}/concpet.txt', 'w') as f:
        for index, values in enumerate(data[0]):
            line = f"{index}: {', '.join(values)}\n"
            f.write(line)

def main():

    args = parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, preprocess = clip.load("ViT-B/32", device=device)

    all_words = []
    all_synsets = []
    base_template = ['A photo of a']
    template = True    
    adaptive = True
    data = args.data_size   # 1k, 20k, 80k, 365, broaden
    layer = args.layer      # fc, l4, l3, l2, l1
    detail = args.detail    # True(minor concept), False(major concept)
    args.img_save_root = f'{args.img_save_root}/example_val_{layer}'

    if template:
        all_words_wotem = []
    else:
        all_words_wotem = None
        adaptive = False

    if data== "1":
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
    elif data == "80":
        nltk.download('wordnet')
        from nltk.corpus import wordnet as wn
        for synset in wn.all_synsets('n'):
            word = synset.lemma_names()[0].replace('_', ' ')
            if template:
                all_words.append(f'A photo of a {word}')
                all_words_wotem.append(f'{word}')
            else:
                all_words.append(f'{word}')
            all_synsets.append(synset)
    elif data == "365":
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
    elif data== 'asd': 
        with open('./utils/part_detail_words.txt', 'r') as f:  # directory of asd.txt
            words = (f.read()).split('\n')
        for word in words:
            if template:
                all_words.append(f'A photo of a {word}')    
                all_words_wotem.append(f'{word}')
            else:
                all_words.append(f'{word}') # len(all_words) = 71

    if not os.path.exists(args.word_save_root):
        word_features = text_to_feature(all_words, model, device, args) # [71, 512]
        template_features = text_to_feature(base_template, model, device, args, template=template) # [1, 1, 512]
    else:
        word_features = load_word_features(args)
        template_features = text_to_feature(base_template, model, device, args, template=template)

    if not os.path.exists(f'{args.img_feat_root}/img_features_{args.num_example}.pkl'):
        img_features = img_to_features(model, device, preprocess, args, detail=detail)
    else:
        img_features = load_img_features(args, detail=detail)

    if not os.path.exists(args.concept_sim_root +'/concept_sim_'+str(args.num_example)+'_'+str(args.concept_sim_num-1)+'.pkl'):
        compute_concept_similarity(img_features, word_features, args, template_features=template_features.squeeze(), detail=detail, adaptive=adaptive)

    if True:   
        concept_discovery(all_words, args, all_words_wotem=all_words_wotem, detail=detail, adaptive=adaptive, data=data, template=template)
    import pickle


if __name__ == '__main__':
    main()


