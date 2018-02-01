# coding=utf-8

"""generator that yields minibatches of data from coco dataset"""

import os
import cv2
import json
import numpy as np
import os.path as osp

def iterate_minibatches(generator, n_minibatches):
    """like generator, but stops after n_minibatches"""
    for i in range(n_minibatches): yield generator.next()

def load_headers_by_categories(fname='../data/COCO/annotations/instances_train2017.json'):
    data = {}

    with open(fname) as f:
        d = json.load(f)

    for obj in d['annotations']:
        if obj['category_id'] not in data:
            data[obj['category_id']] = []

        data[obj['category_id']].append(
            (
                obj['image_id'],
                tuple(np.array(obj['bbox'][:2], dtype=np.int32)),
                tuple(np.array(obj['bbox'][:2], dtype=np.int32) + np.array(obj['bbox'][2:], dtype=np.int32))
            )
        )

    return data

def constrained_sum_sample_pos(n, total):
    """
        Return a randomly chosen list of n positive integers summing to total.
        Each such list is equally likely to occur.
    """

    dividers = sorted(np.random.choice(np.arange(1, total), n - 1))
    return [a - b for a, b in zip(dividers + [total], [0] + dividers)]

def get_image(id, w, h, dir='../data/COCO/train2017'):
    '''
        Find the existing image
    '''

    im_id, p1, p2 = id
    im = cv2.imread(osp.join(dir, str(im_id).zfill(12) + '.jpg'))
    roi = im[p1[1]:p2[1] + 1, p1[0]:p2[0] + 1, :]
    return cv2.resize(roi, (w, h))

class MinibatchGenerator():
    def __init__(self, data, batch_size, w, h, dir='../data/COCO/train2017/', positive='same'):
        self.data = data
        self.batch_size = batch_size
        self.dir = dir
        self.w = w
        self.h = h
        self.positive = positive

        self.positive_size = batch_size // 2
        self.negative_size = batch_size - self.positive_size

    def selector(self, quantities):
        x1 = []
        x2 = []
        y = []

        positive_categories = np.random.choice(self.data.keys(), size=len(quantities))

        for i, cat in enumerate(positive_categories):
            if self.positive == 'same':
                indicies = np.random.choice(np.arange(len(self.data[cat])), size=quantities[i])

                for idx in indicies:
                    img = get_image(self.data[cat][idx], self.w, self.h, self.dir)
                    x1.append(img)
                    x2.append(img)
                    y.append(0)
            elif self.positive == 'class':
                indicies = np.random.choice(np.arange(len(self.data[cat])), size=2*quantities[i])

                for j in range(0, len(indicies), 2):
                    img1 = get_image(self.data[cat][indicies[j]], self.w, self.h, self.dir)
                    img2 = get_image(self.data[cat][indicies[j + 1]], self.w, self.h, self.dir)
                    x1.append(img1)
                    x2.append(img2)
                    y.append(0)
            else:
                raise 'Wrong positive strategy'

        for i in range(self.negative_size):
            cats = np.random.choice(self.data.keys(), size=2, replace=False)
            idx1 = np.random.choice(len(self.data[cats[0]]))
            idx2 = np.random.choice(len(self.data[cats[1]]))
            img1 = get_image(self.data[cats[0]][idx1], self.w, self.h, self.dir)
            img2 = get_image(self.data[cats[1]][idx2], self.w, self.h, self.dir)

            x1.append(img1)
            x2.append(img2)
            y.append(1)

        assert len(x1) == len(y) == len(x2)
        indicies = np.arange(len(x1))
        np.random.shuffle(indicies)

        return np.asarray(x1)[indicies] / 255., np.asarray(x2)[indicies] / 255., np.asarray(y)[indicies].reshape(-1, 1)

    def run(self):
        while True:
            num_positive_categories = np.random.choice(np.arange(1, self.batch_size + 1))
            quantities = constrained_sum_sample_pos(num_positive_categories, self.positive_size)
            yield self.selector(quantities)
