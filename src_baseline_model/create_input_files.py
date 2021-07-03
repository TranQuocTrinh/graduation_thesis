import torch
import numpy as np
import os
import h5py
import json
from scipy.misc import imread, imresize
from tqdm import tqdm
from collections import Counter
from random import seed, choice, sample
from utils import crop_center
import sys


def create_input_files(dataset, data_json_path, image_folder, captions_per_image, min_word_freq, output_folder,
                       max_len=100):
    """
    Creates input files for training, validation, and test data.

    :param dataset: name of dataset, one of 'coco', 'flickr8k', 'flickr30k'
    :param data_json_path: path of data JSON file with splits and captions
    :param image_folder: folder with downloaded images
    :param captions_per_image: number of captions to sample per image
    :param min_word_freq: words occuring less frequently than this threshold are binned as <unk>s
    :param output_folder: folder to save files
    :param max_len: don't sample captions longer than this length
    """

    assert dataset in {'coco', 'flickr8k', 'flickr30k'}

    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Read data JSON
    data = json.load(open(data_json_path, 'r'))

    # Read image paths and captions for each image
    train_image_paths = []
    train_image_captions = []
    val_image_paths = []
    val_image_captions = []
    test_image_paths = []
    test_image_captions = []
    # word_freq = Counter()

    for img in data['images']:
        captions = []
        for c in img['sentences']:
            # Update word frequency
            # word_freq.update(c['tokens'])
            if len(c['tokens']) <= max_len:
                captions.append(c['tokens'])

        if len(captions) == 0:
            continue

        if dataset == 'coco':
            path_img = os.path.join(image_folder, img['filepath'], img['filename'])
        else:
            path_img = os.path.join(image_folder, img['filename'])

        if img['split'] in {'train', 'restval'}:
            train_image_paths.append(path_img)
            train_image_captions.append(captions)
        elif img['split'] in {'val'}:
            val_image_paths.append(path_img)
            val_image_captions.append(captions)
        elif img['split'] in {'test'}:
            test_image_paths.append(path_img)
            test_image_captions.append(captions)

    # Sanity check
    assert len(train_image_paths) == len(train_image_captions)
    assert len(val_image_paths) == len(val_image_captions)
    assert len(test_image_paths) == len(test_image_captions)

    # Create word map
    # words = [w for w in word_freq.keys() if word_freq[w] > min_word_freq]
    # word_map = {k: v + 1 for v, k in enumerate(words)}
    # word_map['<unk>'] = len(word_map) + 1
    # word_map['<start>'] = len(word_map) + 1
    # word_map['<end>'] = len(word_map) + 1
    # word_map['<pad>'] = 0

    word_map = json.load(open('../data_processing_first_step/WORDMAP.json'))

    # Create a base/root name for all output files
    base_filename = dataset + '_' + str(captions_per_image) + '_cap_per_img_' + str(min_word_freq) + '_min_word_freq'

    # Save word map to a JSON
    path = os.path.join(output_folder, 'WORDMAP_' + base_filename + '.json')
    json.dump(word_map, open(path, 'w'))

    # Sample captions for each image, save images to HDF5 file, and captions and their lengths to JSON files
    seed(123)
    for impaths, imcaps, split in [(train_image_paths, train_image_captions, 'TRAIN'),
                                   (val_image_paths, val_image_captions, 'VAL'),
                                   (test_image_paths, test_image_captions, 'TEST')]:

        with h5py.File(os.path.join(output_folder, split + '_IMAGES_' + base_filename + '.hdf5'), 'a') as h:
            # Make a note of the number of captions we are sampling per image
            h.attrs['captions_per_image'] = captions_per_image

            # Create dataset inside HDF5 file to store images
            images = h.create_dataset('images', (len(impaths), 3, 224, 224), dtype='uint8')

            print("\nReading %s images and captions, storing to file...\n" % split)

            enc_captions = []
            caplens = []

            for i, path in enumerate(tqdm(impaths)):

                # Sample captions
                if len(imcaps[i]) < captions_per_image:
                    captions = imcaps[i] + [choice(imcaps[i]) for _ in range(captions_per_image - len(imcaps[i]))]
                else:
                    captions = sample(imcaps[i], k=captions_per_image)

                # Sanity check
                assert len(captions) == captions_per_image

                # Read images
                img = imread(impaths[i])
                if len(img.shape) == 2:
                    img = img[:, :, np.newaxis]
                    img = np.concatenate([img, img, img], axis=2)
                img = imresize(img, (256, 256))
                img = img.transpose(2, 0, 1)    
                # center crop
                img = crop_center(img, 224, 224)

                assert img.shape == (3, 224, 224)
                assert np.max(img) <= 255

                # Save image to HDF5 file
                images[i] = img

                for j, c in enumerate(captions):
                    # Encode captions
                    enc_c = [word_map['<start>']] + [word_map.get(word, word_map['<unk>']) for word in c] + [
                        word_map['<end>']] + [word_map['<pad>']] * (max_len - len(c))

                    # Find caption lengths
                    c_len = len(c) + 2

                    enc_captions.append(enc_c)
                    caplens.append(c_len)

            # Sanity check
            assert images.shape[0] * captions_per_image == len(enc_captions) == len(caplens)

            # Save encoded captions and their lengths to JSON files
            path = os.path.join(output_folder, split + '_CAPTIONS_' + base_filename + '.json')
            json.dump(enc_captions, open(path, 'w'))

            path = os.path.join(output_folder, split + '_CAPLENS_' + base_filename + '.json')
            json.dump(caplens, open(path, 'w'))


def main():
    dataset = os.getcwd().split('/')[-2]
    if dataset == 'coco':
        img_fol = '../original/'
    else:
        img_fol = '../original/' + dataset + '_images/'

    create_input_files(dataset=dataset,
                       data_json_path='../original/dataset_' + dataset + '.json',
                       image_folder=img_fol,
                       captions_per_image=5,
                       min_word_freq=5,
                       output_folder='../data_processing_first_step/',
                       max_len=50)


if __name__ == '__main__':
    main()
