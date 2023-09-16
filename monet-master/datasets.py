# License: MIT
# Author: Karl Stelzner

import os
import sys

import torch
from torch.utils.data import Dataset
import numpy as np
from numpy.random import random_integers
from PIL import Image
import h5py
import cv2


def progress_bar(count, total, status=''):
    bar_len = 60
    filled_len = int(round(bar_len * count / float(total)))

    percents = round(100.0 * count / float(total), 1)
    bar = '=' * filled_len + '-' * (bar_len - filled_len)

    sys.stdout.write('[%s] %s%s ...%s\r' % (bar, percents, '%', status))
    sys.stdout.flush()

def make_sprites(n=50000, height=64, width=64):
    images = np.zeros((n, height, width, 3))
    counts = np.zeros((n,))
    print('Generating sprite dataset...')
    for i in range(n):
        num_sprites = random_integers(0, 2)
        counts[i] = num_sprites
        for j in range(num_sprites):
            pos_y = random_integers(0, height - 12)
            pos_x = random_integers(0, width - 12)

            scale = random_integers(12, min(16, height-pos_y, width-pos_x))

            cat = random_integers(0, 2)
            sprite = np.zeros((height, width, 3))

            if cat == 0:  # draw circle
                center_x = pos_x + scale // 2.0
                center_y = pos_y + scale // 2.0
                for x in range(height):
                    for y in range(width):
                        dist_center_sq = (x - center_x)**2 + (y - center_y)**2
                        if  dist_center_sq < (scale // 2.0)**2:
                            sprite[x][y][cat] = 1.0
            elif cat == 1:  # draw square
                sprite[pos_x:pos_x + scale, pos_y:pos_y + scale, cat] = 1.0
            else:  # draw square turned by 45 degrees
                center_x = pos_x + scale // 2.0
                center_y = pos_y + scale // 2.0
                for x in range(height):
                    for y in range(width):
                        if abs(x - center_x) + abs(y - center_y) < (scale // 2.0):
                            sprite[x][y][cat] = 1.0
            images[i] += sprite
        if i % 100 == 0:
            progress_bar(i, n)
    images = np.clip(images, 0.0, 1.0)

    return {'x_train': images[:4 * n // 5],
            'count_train': counts[:4 * n // 5],
            'x_test': images[4 * n // 5:],
            'count_test': counts[4 * n // 5:]}


class Sprites(Dataset):
    def __init__(self, directory, n=50000, canvas_size=64,
                 train=True, transform=None):
        # np_file = 'sprites_{}_{}.npz'.format(n, canvas_size)
        # h5py_file = 'mini_circle_cont_data.h5'
        # h5py_file = 'mini_circle_diff_cont_data.h5'
        h5py_file = 'mini_circle_2data.h5'
        # h5py_file = 'sprite_data.h5'
        full_path = os.path.join(directory, h5py_file)
        hdf5_file = h5py.File(full_path, 'r')
        # if not os.path.isfile(full_path):
        #     gen_data = make_sprites(n, canvas_size, canvas_size)
        #     np.savez(np_file, **gen_data)

        # data = np.load(full_path)

        self.transform = transform
        self.images = np.array(hdf5_file['training']['features']) if train else np.array(hdf5_file['validation']['features'])
        self.actions = np.array(hdf5_file['training']['actions']) if train else np.array(
            hdf5_file['validation']['actions'])  # (T, s, 2)
        self.images = np.transpose(self.images, (1, 0, 2, 3, 4))
        self.actions = np.transpose(self.actions, (1, 0, 2))
        self.counts = 5 * np.ones_like(self.actions) if train else 11 * np.ones_like(self.actions)

    def __len__(self):
        return self.images.shape[0]

    def __getitem__(self, idx):
        # img = self.images[idx][0]
        img = cv2.resize(self.images[idx][0], dsize=(32, 32), interpolation=cv2.INTER_CUBIC)
        if self.transform is not None:
            img = self.transform(img)
        return img, self.counts[idx][0][0]


class Cobra(Dataset):
    def __init__(self, directory, n=50000, canvas_size=64,
                 train=True, transform=None):
        # np_file = 'sprites_{}_{}.npz'.format(n, canvas_size)
        # h5py_file = 'mini_circle_cont_data.h5'
        # h5py_file = 'mini_circle_diff_cont_data.h5'

        # h5py_file = 'sprite_data.h5'
        h5py_file = 'mini_circle_2data.h5'
        full_path = os.path.join(directory, h5py_file)
        hdf5_file = h5py.File(full_path, 'r')
        # if not os.path.isfile(full_path):
        #     gen_data = make_sprites(n, canvas_size, canvas_size)
        #     np.savez(h5py_file, **gen_data)

        # data = np.load(full_path)

        self.transform = transform
        # self.images = data['x_train'] if train else data['x_test']
        self.images = np.array(hdf5_file['training']['features']) if train else np.array(hdf5_file['validation']['features']) # (T,s, 64, 64, 3)
        self.actions = np.array(hdf5_file['training']['actions']) if train else np.array(hdf5_file['validation']['actions']) # (T,s, 2)
        self.images = np.transpose(self.images, (1, 0, 2, 3, 4))
        self.actions = np.transpose(self.actions, (1, 0, 2))

        self.counts = 5 * np.ones_like(self.actions) if train else 11 * np.ones_like(self.actions)


        # print('counts_shape:', self.counts[0][0].shape)
        # print('counts_shape11:',self.counts[0][0][0])



    def __len__(self):
        return self.images.shape[0]

    def __getitem__(self, idx):
        # img = self.images[idx][0]
        # next_img = self.images[idx][1]
        img = cv2.resize(self.images[idx][0], dsize=(32, 32), interpolation=cv2.INTER_CUBIC)
        next_img = cv2.resize(self.images[idx][1], dsize=(32, 32), interpolation=cv2.INTER_CUBIC)
        action = self.actions[idx][0]  # shape : (2,)
        if self.transform is not None:
            img = self.transform(img)
            next_img = self.transform(next_img)

        return img, next_img, action, self.counts[idx][0][0]


class Clevr(Dataset):
    def __init__(self, directory, transform=None):
        self.directory = directory
        self.filenames = os.listdir(directory)
        self.n = len(self.filenames)
        self.transform = transform

    def __len__(self):
        return self.n

    def __getitem__(self, idx):
        imgpath = os.path.join(self.directory, self.filenames[idx])
        img = Image.open(imgpath)
        if self.transform is not None:
            img = self.transform(img)
        return img, 1

