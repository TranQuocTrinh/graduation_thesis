from tqdm import tqdm
import json

coco = json.load(open('../../coco/original/dataset_coco.json'))
k8 = json.load(open('../../flickr8k/original/dataset_flickr8k.json'))
k30 = json.load(open('../../flickr30k/original/dataset_flickr30k.json'))

all_tokens = set()
dct = {}
    for img in tqdm(coco['images']+k8['images']+k30['images']):
        for sen in img['sentences']:
            for token in sen['tokens']:
                l = len(all_tokens)
                all_tokens.add(token)
                if len(all_tokens) > l:
                    dct[token] = 0
                else:
                    dct[token] += 1

word_map = {k:v+1 for v,k in enumerate(dct.keys()) if dct[k] > 7}
word_map['<unk>'] = len(word_map) + 1
word_map['<start>'] = len(word_map) + 1
word_map['<end>'] = len(word_map) + 1
word_map['<pad>'] = 0

json.dump(word_map, open('../data_processing_first_step/WORDMAP.json','w'))
