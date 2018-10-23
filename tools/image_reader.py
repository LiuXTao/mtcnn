import cv2
import numpy as np
import random

class TrainImageReader(object):
    def __init__(self, imdb, im_size, batch_size=256, shuffle=False):
        self.imdb = imdb
        self.batch_size = batch_size
        self.im_size = im_size
        self.shuffle = shuffle

        self.cur = 0
        self.size = len(imdb)
        self.index = np.arange(self.size)
        self.num_classes = 2

        self.batch = None
        self.data = None
        self.label = None

        self.label_names = ['label', 'bbox_target', 'landmark_target']

    def reset(self):
        self.cur = 0
        if self.shuffle:
            np.random.shuffle(self.index)

    def iter_next(self):
        return self.cur < self.size

    def __iter__(self):
        return self

    def next(self):
        if self.iter_next():
            self.get_batch()
            self.cur = min(self.cur + self.batch_size, self.size)
            return self.data, self.label
        else:
            raise StopIteration

    def __next__(self):
        return self.next()


    def get_batch(self):
        cur_from = self.cur
        cur_to = min(self.cur + self.batch_size, self.size)
        imdb = [self.imdb[self.index[i]] for i in range(cur_from, cur_to)]
        data, label = get_min_batch(imdb)

        self.data = data['data']
        self.label = [label[name] for name in self.label_names]

    def get_index(self):
        return self.cur / self.batch_size

    def get_pad(self):
        if self.cur + self.batch_size > self.size:
            return self.batch_size + self.cur - self.size
        else:
            return 0

def get_min_batch(imdb):
    num_images = len(imdb)
    processed_imd = list()
    cls_label = list()
    bbox_reg_target = list()
    landmark_reg_target = list()

    for i in range(num_images):
        im = cv2.imread(imdb[i]['image'])
        if imdb[i]['flipped']:
            im = im[:, ::-1, :]

        cls = imdb[i]['label']
        bbox_target = imdb[i]['bbox_target']
        landmark = imdb[i]['landmark_target']

        processed_imd.append(im)
        cls_label.append(cls)
        bbox_reg_target.append(bbox_target)
        landmark_reg_target.append(landmark)

    processed_imd = np.asarray(processed_imd)
    cls_label = np.asarray(cls_label)
    bbox_reg_target = np.asarray(bbox_reg_target)
    landmark_reg_target = np.asarray(landmark_reg_target)

    data = {'data': processed_imd}
    label = {
        'label': cls_label,
        'bbox_target': bbox_reg_target,
        'landmark_target': landmark_reg_target
    }

    return data, label

class TestImageReader(object):
    def __init__(self, imdb, batch_size=1, shuffle=False):
        self.imdb = imdb
        self.batch_size = batch_size

        self.shuffle = shuffle

        self.cur = 0
        self.size = len(imdb)
        self.index = np.arange(self.size)

        self.data = None
        self.label = None



    def reset(self):
        self.cur = 0
        if self.shuffle:
            np.random.shuffle(self.index)

    def iter_next(self):
        return self.cur + self.batch_size <= self.size

    def __iter__(self):
        return self

    def next(self):
        if self.iter_next():
            self.get_batch()
            self.cur += self.batch_size
            return self.data
        else:
            raise StopIteration

    def __next__(self):
        return self.next()


    def get_batch(self):
        cur_from = self.cur
        cur_to = min(self.cur + self.batch_size, self.size)
        imdb = [self.imdb[self.index[i]] for i in range(cur_from, cur_to)]
        data = get_test_batch(imdb)

        self.data = data['data']


    def get_index(self):
        return self.cur / self.batch_size

    def get_pad(self):
        if self.cur + self.batch_size > self.size:
            return self.batch_size + self.cur - self.size
        else:
            return 0

def get_test_batch(imdb):
    assert len(imdb) == 1, 'Single batch only'
    i = random.sample(range(0, len(imdb)), 1)[0]
    im = cv2.imread(imdb[i]['image'])
    data = {'data': im}
    return data