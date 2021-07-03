import torch
import torch.nn.functional as F
import numpy as np
import json
import torchvision.transforms as transforms
import skimage.transform
import argparse
from scipy.misc import imread, imresize
from PIL import Image
from tqdm import tqdm
import sys
from utils import *
import os

dataset = os.getcwd().split('/')[-2]

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def caption_image_beam_search(encoder, decoder, image_path, word_map, beam_size=3):
    """
    Reads an image and captions it with beam search.

    :param encoder: encoder model
    :param decoder: decoder model
    :param image_path: path to image
    :param word_map: word map
    :param beam_size: number of sequences to consider at each decode-step
    :return: caption, weights for visualization
    """

    k = beam_size
    vocab_size = len(word_map)

    # Read image and process
    img = imread(image_path)
    if len(img.shape) == 2:
        img = img[:, :, np.newaxis]
        img = np.concatenate([img, img, img], axis=2)
    img = imresize(img, (256, 256))
    img = img.transpose(2, 0, 1)
    img = crop_center(img, 224, 224)
    img = img / 255.

    img = torch.FloatTensor(img).to(device)
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    transform = transforms.Compose([normalize])
    image = transform(img)  # (3, 256, 256)

    # Encode
    image = image.unsqueeze(0)  # (1, 3, 256, 256)
    encoder_out = encoder(image)  # (1, enc_image_size, enc_image_size, encoder_dim)
    enc_image_size = encoder_out.size(1)
    encoder_dim = encoder_out.size(3)

    # Flatten encoding
    encoder_out = encoder_out.view(1, -1, encoder_dim)  # (1, num_pixels, encoder_dim)
    num_pixels = encoder_out.size(1)

    # We'll treat the problem as having a batch size of k
    encoder_out = encoder_out.expand(k, num_pixels, encoder_dim)  # (k, num_pixels, encoder_dim)

    # Tensor to store top k previous words at each step; now they're just <start>
    k_prev_words = torch.LongTensor([[word_map['<start>']]] * k).to(device)  # (k, 1)

    # Tensor to store top k sequences; now they're just <start>
    seqs = k_prev_words  # (k, 1)

    # Tensor to store top k sequences' scores; now they're just 0
    top_k_scores = torch.zeros(k, 1).to(device)  # (k, 1)

    # Lists to store completed sequences, their scores
    complete_seqs = list()
    complete_seqs_scores = list()

    # Start decoding
    step = 1
    h, c = decoder.init_hidden_state(encoder_out)

    # s is a number less than or equal to k, because sequences are removed from this process once they hit <end>
    while True:

        embeddings = decoder.embedding(k_prev_words).squeeze(1)  # (s, embed_dim)

        h, c = decoder.decode_step(embeddings, (h, c))  # (s, decoder_dim)

        scores = decoder.fc(h)  # (s, vocab_size)
        scores = F.log_softmax(scores, dim=1)

        # Add
        scores = top_k_scores.expand_as(scores) + scores  # (s, vocab_size)

        # For the first step, all k points will have the same scores (since same k previous words, h, c)
        if step == 1:
            top_k_scores, top_k_words = scores[0].topk(k, 0, True, True)  # (s)
        else:
            # Unroll and find top scores, and their unrolled indices
            top_k_scores, top_k_words = scores.view(-1).topk(k, 0, True, True)  # (s)

        # Convert unrolled indices to actual indices of scores
        prev_word_inds = top_k_words / vocab_size  # (s)
        next_word_inds = top_k_words % vocab_size  # (s)

        # Add new words to sequences
        seqs = torch.cat([seqs[prev_word_inds], next_word_inds.unsqueeze(1)], dim=1)  # (s, step+1)


        # Which sequences are incomplete (didn't reach <end>)?
        incomplete_inds = [ind for ind, next_word in enumerate(next_word_inds) if
                           next_word != word_map['<end>']]
        complete_inds = list(set(range(len(next_word_inds))) - set(incomplete_inds))

        # Set aside complete sequences
        if len(complete_inds) > 0:
            complete_seqs.extend(seqs[complete_inds].tolist())
            complete_seqs_scores.extend(top_k_scores[complete_inds])
        k -= len(complete_inds)  # reduce beam length accordingly

        # Proceed with incomplete sequences
        if k == 0:
            break
        seqs = seqs[incomplete_inds]
        h = h[prev_word_inds[incomplete_inds]]
        c = c[prev_word_inds[incomplete_inds]]
        encoder_out = encoder_out[prev_word_inds[incomplete_inds]]
        top_k_scores = top_k_scores[incomplete_inds].unsqueeze(1)
        k_prev_words = next_word_inds[incomplete_inds].unsqueeze(1)

        # Break if things have been going on too long
        if step > 50:
            break
        step += 1

    i = complete_seqs_scores.index(max(complete_seqs_scores))
    seq = complete_seqs[i]

    return seq


if __name__ == '__main__':
    name_model = sys.argv[2]

    dt = json.load(open('/data/tqtrinh/'+dataset+'/original/dataset_' + dataset +'.json'))
    if dataset == 'coco':
        test_images = [image['filename'] for image in dt['images'] if image['split']=='test']
        img_path = ['/data/tqtrinh/'+dataset+'/original/' + image['filepath'] +'/' + image['filename'] for image in dt['images'] if image['split'] == 'test']
    else:
        test_images = [image['filename'] for image in dt['images'] if image['split'] == 'test']
        img_path = ['/data/tqtrinh/'+dataset+'/original/'+ dataset + '_images/' + image for image in test_images]

    path_model = '../checkpoint/BEST_checkpoint_' + dataset + '_5_cap_per_img_5_min_word_freq.pth.tar'
    word_map_file = '/data/tqtrinh/'+dataset+'/data_processing_first_step/WORDMAP_' + dataset + '_5_cap_per_img_5_min_word_freq.json'
    beam_size = int(sys.argv[1])
    dct_name_id = json.load(open('/data/tqtrinh/'+dataset+'/original/dct_image_id_' + dataset + '.json'))


    # Load model
    checkpoint = torch.load(path_model, map_location=str(device))
    decoder = checkpoint['decoder']
    decoder = decoder.to(device)
    decoder.eval()
    encoder = checkpoint['encoder']
    encoder = encoder.to(device)
    encoder.eval()

    # Load word map (word2ix)
    with open(word_map_file, 'r') as j:
        word_map = json.load(j)
    rev_word_map = {v: k for k, v in word_map.items()}  # ix2word

    results = []

    for image,path in tqdm(zip(test_images, img_path), total=len(test_images)):
        #img = image
        # Encode, decode with attention and beam search
        seq = caption_image_beam_search(encoder, decoder, path, word_map, beam_size)
        results.append({'image_id':dct_name_id[image], 'caption':' '.join([rev_word_map[ind] for ind in seq][1:-1])})
        print(results[-1])
    json.dump(results, open('/data/tqtrinh/'+dataset+'/evaluate/coco-caption/results/results-'+ name_model+'-' + dataset + '-' + str(beam_size) +'.json','w'))
