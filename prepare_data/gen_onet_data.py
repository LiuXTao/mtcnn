import sys
import numpy as np
import cv2
import os
import pickle
import time
import numpy.random as npr
from tools.utils import IoU
import tools.config as config
import tools.visual_tools as vision
from tools.utils import convert_to_square, IoU
from tools.imagedb import ImageDB
from tools.image_reader import TestImageReader
from tools.detect_utils import create_mtcnn_net
from train.MtcnnDetector import MtcnnDector

def gen_onent_data(data_dir, anno_file, pnet_model_file, rnet_model_file, prefix_path='', use_cuda=False, vis=False):
    pnet, rnet, _ = create_mtcnn_net(pnet_model_file, rnet_model_file, use_cuda=use_cuda)
    mtcnn_detector = MtcnnDector(pnet=pnet, rnet=rnet, min_face_size=12)

    imagedb = ImageDB(anno_file, mode='test', prefix_path=prefix_path)
    imdb = imagedb.load_imdb()
    image_reader = TestImageReader(imdb, 1, False)

    all_boxes = list()
    batch_idx = 0

    for data_batch in image_reader:
        if data_batch % 100 == 0:
            print("%d images done" % batch_idx)

        im = data_batch
        t = time.time()

        p_boxes, p_boxes_align = mtcnn_detector.detect_pnet(im)
        boxes, boxes_align = mtcnn_detector.detect_rnet(im=im, dets=p_boxes_align)

        if boxes_align is None:
            all_boxes.append(np.array([]))
            batch_idx += 1
            continue

        if vis:
            im_tmp = cv2.cvtColor(np.asarray(im), cv2.COLOR_RGB2BGR)
            vision.visual_two(im_tmp, boxes, boxes_align)

        t1 = time.time() - t
        print('time cost for image ', batch_idx, '/', image_reader.size, ': ', t1)
        all_boxes.append(boxes_align)
        batch_idx += 1

    save_path = config.MODLE_STORE_DIR

    if not os.path.exists(save_path):
        os.makedirs(save_path)

    save_file = os.path.join(save_path, 'rnet_detections_%d.pkl'%int(time.time()))

    with open(save_file, 'wb') as f:
        pickle.dump(all_boxes, f, pickle.HIGHEST_PROTOCOL)

    gen_onet_sample_data(data_dir, anno_file, save_file, prefix_path)

def gen_onet_sample_data(data_dir, anno_file, save_file, prefix):

    neg_save_dir = os.path.join(data_dir, '48/negative')
    pos_save_dir = os.path.join(data_dir, '48/positive')
    part_save_dir = os.path.join(data_dir, '48/part')

    for dir_path in [neg_save_dir, pos_save_dir, part_save_dir]:
        if not os._exists(dir_path):
            os.makedirs(dir_path)

    with open(anno_file, 'r') as f:
        annotations = f.readlines()


    num_of_image = len(annotations)
    im_idx_list = list()
    gt_boxes_list = list()
    image_size = 48

    for annotation in annotations:
        annotation = annotation.strip().split(' ')
        im_idx = os.path.join(prefix, annotation[0])

        boxes = map(float, annotation[1:])
        boxes = np.array(boxes, dtype=np.float32).reshape(-1, 4)
        im_idx_list.append(im_idx)
        gt_boxes_list.append(boxes)

    save_path = config.ANNO_STORE_DIR
    if not os.exist(save_path):
        os.makedirs(save_path)

    f1 = open(os.path.join(save_path, 'pos_%d.txt'%image_size), 'w')
    f2 = open(os.path.join(save_path, 'neg_%d.txt'%image_size), 'w')
    f3 = open(os.path.join(save_path, 'part_%d.txt'%image_size), 'w')

    det_h = open(save_file, 'r')
    det_boxes = pickle.load(save_file)
    print(len(det_boxes), num_of_image)

    assert len(det_boxes) == num_of_image, 'wrong detection or ground truths'

    n_idx = 0
    p_idx = 0
    d_idx = 0
    image_done = 0

    for im_idx, dets, gts in zip(im_idx_list, det_boxes, gt_boxes_list):
        image_done += 1

        if image_done % 100 == 0:
            print("%d images done" % image_done)
        if dets.shape[0] == 0:
            continue

        img = cv2.imread(im_idx)
        dets = convert_to_square(dets)
        dets[:, 0:4] = np.round(dets[:, 0:4])

        # each image have at most 50 neg_samples
        cur_n_idx = 0
        for box in dets:
            x_left, y_top, x_right, y_bottom = box[0:4].astype(int)
            w, h = x_right - x_left, y_bottom - y_top

            if w < 20 or x_left <= 0 or y_top <= 0 or x_right >= img.shap[1] or y_bottom >= img.shape[0]:
                continue

            iou = IoU(box, gts)
            cropped_im = img[y_top: y_bottom + 1, x_left: x_right + 1, :]
            resized_im = cv2.resize(cropped_im, (image_size, image_size), interpolation=cv2.INTER_LINEAR)

            if np.max(iou) < 0.3:
                cur_n_idx += 1
                if cur_n_idx <= 50:
                    save_file = os.path.join(neg_save_dir, '%s.jpg' % n_idx)
                    f2.write(save_file + '0\n')
                    cv2.imwrite(save_file, resized_im)
                    n_idx += 1

            else:
                idx = np.argmax(iou)
                gt_tmp = gts[idx]
                x1, y1, x2, y2 = gt_tmp

                off_x1, off_y1, off_x2, off_y2 = (x1 - x_left) / float(w), (y1 - y_top) / float(h), (
                            x2 - x_right) / float(w), (y2 - y_bottom) / float(h)
                if np.max(iou) >= 0.65:
                    save_file = os.path.join(pos_save_dir, "%s.jpg" % p_idx)
                    f1.write(save_file + '1 %.2f %.2f %.2f %.2f\n' % (off_x1, off_y1, off_x2, off_y2))
                    cv2.imread(save_file, resized_im)
                    p_idx += 1

                elif np.max(iou) >= 0.4:
                    save_file = os.path.join(part_save_dir, "%s.jpg" % d_idx)
                    f3.write(save_file, '-1 %.2f %.2f %.2f %.2f\n' % (off_x1, off_y1, off_x2, off_y2))
                    cv2.imread(save_file, resized_im)
                    d_idx += 1

        f1.close()
        f2.close()
        f3.close()


if __name__ == '__main__':

    data_dir = config.TRAIN_DATA_DIR
    anno_file = os.path.join(config.ANNO_STORE_DIR, "wider_origin_anno.txt")
    pnet_path = os.path.join(config.MODLE_STORE_DIR, "pnet_model_final.pt")
    rnet_path = os.path.join(config.MODLE_STORE_DIR, "rnet_model_final.pt")
    prefix_path = "D://data/Widerface/WIDER_train/images"

    gen_onent_data(data_dir, anno_file, pnet_path, rnet_path, prefix_path)









