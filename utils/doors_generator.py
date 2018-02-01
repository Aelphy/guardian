# coding=utf-8

"""generator that yields minibatches of data from doors dataset"""

import os
import cv2
import sys
import json
import numpy as np
import os.path as osp

sys.path.append('../share/src/door_state_midlevel_fusion/')

from load_door_data import load_doors

def test_iterate_minibatches(test_img, other_images, batch_size):
    for i in range(0, len(other_images), batch_size):
        a = other_images[i : i + batch_size]
        b = np.tile(test_img, (len(a), 1, 1, 1))

        if len(a) > 0:
            yield np.asarray(a)/255., np.asarray(b)/255.

def iterate_minibatches(generator, n_minibatches):
    """like generator, but stops after n_minibatches"""
    for i in range(n_minibatches): yield generator.next()

def load_data(img_cols=256, img_rows=256, TRAIN_DATA='/scratch/home/aelphy/projects/guardian/share/docs/', IMG_FOLDER='/scratch/home/aelphy/projects/guardian/data/doors/'):
    Xn, yn, wpn = load_doors(img_rows, img_cols, '', 'new_doors', train_data=TRAIN_DATA, img_folder=IMG_FOLDER)
    Xn_depth, yn, wpn_depths = load_doors(img_rows, img_cols, '','new_depth', train_data=TRAIN_DATA, img_folder=IMG_FOLDER)

    Xn1, yn1, wpn1 = load_doors(img_rows, img_cols, '', 'new_doors_20171109', train_data=TRAIN_DATA, img_folder=IMG_FOLDER)
    Xn_depth1, yn1, wpn_depths1 = load_doors(img_rows, img_cols, '','new_depth_20171109', train_data=TRAIN_DATA, img_folder=IMG_FOLDER)

    Xs, ys, wps = load_doors(img_rows, img_cols, '', 'swing_doors', train_data=TRAIN_DATA, img_folder=IMG_FOLDER)
    Xs_depth, ys, wps_depths = load_doors(img_rows, img_cols, '','swing_depth', train_data=TRAIN_DATA, img_folder=IMG_FOLDER)

    Xr, yr, wpr = load_doors(img_rows, img_cols, '', 'rollup_doors', train_data=TRAIN_DATA, img_folder=IMG_FOLDER)
    Xr_depth, yr, wpr_depths = load_doors(img_rows, img_cols, '','rollup_depth', train_data=TRAIN_DATA, img_folder=IMG_FOLDER)

    return np.vstack([Xs, Xr, Xn, Xn1]), np.vstack([Xs_depth, Xr_depth, Xn_depth, Xn_depth1]), np.argmax(np.vstack([ys, yr, yn, yn1]), axis=1), np.hstack([wps, wpr, wpn, wpn1])

class PairMinibatchGenerator():
    def __init__(self, open_doors, closed_doors, hand_crafted_dir, hand_crafted_filepath, batch_size):
        self.open_doors = open_doors
        self.closed_doors = closed_doors
        self.open_reference_doors = []
        self.closed_reference_doors = []

        w, h, _ = open_doors[0].shape

        with open(hand_crafted_filepath) as f:
            for line in f:
                parts = line.strip().split(', ')
                im = cv2.resize(cv2.imread(osp.join(hand_crafted_dir, parts[0])), (w, h))

                if int(parts[1]) == 1:
                    self.open_reference_doors.append(im)
                else:
                    self.closed_reference_doors.append(im)

        self.batch_size = batch_size

        self.positive_size = batch_size // 2
        self.negative_size = batch_size - self.positive_size

    def selector(self):
        x1 = []
        x2 = []
        y = []

        idxs_open = np.random.choice(len(self.open_doors), size=self.batch_size, replace=False)
        idxs_closed = np.random.choice(len(self.closed_doors), size=self.batch_size, replace=False)

        reference_idxs_open = np.random.choice(len(self.open_reference_doors), size=self.batch_size // 2, replace=True)
        reference_idxs_closed = np.random.choice(len(self.closed_reference_doors), size=self.batch_size // 2, replace=True)

        for i in range(self.positive_size // 2):
            x1.append(self.open_doors[idxs_open[i]])
            x2.append(self.open_reference_doors[reference_idxs_open[i]])
            y.append(0)
            x1.append(self.closed_doors[idxs_closed[i]])
            x2.append(self.closed_reference_doors[reference_idxs_closed[i]])
            y.append(0)

        for i in range(self.negative_size // 2):
            x1.append(self.open_doors[idxs_open[i + self.positive_size]])
            x2.append(self.closed_reference_doors[reference_idxs_closed[i + self.positive_size // 2]])
            y.append(1)
            x1.append(self.closed_doors[idxs_open[i + self.positive_size]])
            x2.append(self.open_reference_doors[reference_idxs_open[i + self.positive_size // 2]])
            y.append(1)

        assert len(x1) == len(y) == len(x2)
        indicies = np.arange(len(x1))
        np.random.shuffle(indicies)

        return np.asarray(x1)[indicies] / 255., np.asarray(x2)[indicies] / 255., np.asarray(y)[indicies].reshape(-1, 1)

    def run(self):
        while True:
            yield self.selector()

class MinibatchGenerator():
    def __init__(self, open_doors, closed_doors, batch_size):
        self.open_doors = open_doors
        self.closed_doors = closed_doors
        self.batch_size = batch_size

        self.positive_size = batch_size // 2
        self.negative_size = batch_size - self.positive_size

    def selector(self):
        x1 = []
        x2 = []
        y = []

        idxs_open = np.random.choice(len(self.open_doors), size=self.positive_size, replace=False)
        idxs_closed = np.random.choice(len(self.closed_doors), size=self.positive_size, replace=False)

        for i in range(self.positive_size // 2):
            x1.append(self.open_doors[idxs_open[i]])
            x2.append(self.open_doors[idxs_open[i + self.positive_size // 2]])
            y.append(0)
            x1.append(self.closed_doors[idxs_closed[i]])
            x2.append(self.closed_doors[idxs_closed[i + self.positive_size // 2]])
            y.append(0)

        for i in range(self.negative_size):
            idx1 = np.random.choice(len(self.open_doors))
            idx2 = np.random.choice(len(self.closed_doors))

            x1.append(self.open_doors[idx1])
            x2.append(self.closed_doors[idx2])
            y.append(1)

        assert len(x1) == len(y) == len(x2)
        indicies = np.arange(len(x1))
        np.random.shuffle(indicies)

        return np.asarray(x1)[indicies] / 255., np.asarray(x2)[indicies] / 255., np.asarray(y)[indicies].reshape(-1, 1)

    def run(self):
        while True:
            yield self.selector()
