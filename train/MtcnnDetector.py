import cv2
import sys
import os
import time
import numpy as np
import torch

from nets.models import PNet, RNet, ONet
import tools.utils as utils
import tools.image_tools as image_tools

class MtcnnDector(object):
    ''' P, R, O net for face detection and landmark alignment'''
    def __init__(self, pnet=None, rnet=None, onet=None,
                 min_face_size=12, stride=2, threshold=[0.6, 0.7, 0.7], scale_factor=0.709):
        self.pnet_dector = pnet
        self.rnet_dector = rnet
        self.onet_dector = onet
        self.min_face_size = min_face_size
        self.stride = stride
        self.threshold = threshold
        self.scale_factor = scale_factor

    def unique_image_format(self, im):
        if not isinstance(im, np.ndarray):
            if im.mode == 'I':
                im = np.array(im, np.int32, copy=False)
            elif im.mode == 'I;16':
                im = np.array(im, np.int16, copy=False)
            else:
                im = np.ndarray(im)

        return im

    def square_box(self, bbox):
        '''
        convert bbox to square
        Parameters:
            bbox: numpy array, shape n x m
        Returns:
            square bbox
        '''
        square_box = bbox.copy()
        x1, y1, x2, y2 = [bbox[:, i] for i in range(4)]
        h = y2 - y1 + 1.0
        w = x2 - x1 + 1.0

        size = np.maximum(h, w)
        square_box[:, 0] = x1 + w*0.5 - size*0.5
        square_box[:, 1] = y1 + h*0.5 - size*0.5
        square_box[:, 2] = square_box[:, 0] + size - 1
        square_box[:, 3] = square_box[:, 1] + size - 1

        return square_box

    def generate_bounding_box(self, maps, reg, scale, threshold):
        '''
        generate bbox from feature map
        for PNet, there exists no fc layer, only convolution layer ,so feature map n x m x 1/4
        Parameters:
            maps: numpy array , n x m x 1, detect score for each position
            reg: numpy array , n x m x 4, bbox
            scale: float number, scale of this detection
            threshold: float number, detect threshold
        Returns:
            bbox array
        '''
        stride = 2
        cell_size = 12
        t_index = np.where(maps > threshold)
        if t_index[0].size == 0:
            return np.array([])

        dx1, dy1, dx2, dy2 = [reg[0, t_index[0], t_index[1], i] for i in range(4)]
        offsets = np.array([dx1, dy1, dx2, dy2])

        score = maps[t_index[0], t_index[1], 0]
        bounding_box = np.vstack(
            np.round((stride * t_index[1]) / scale),
            np.round((stride * t_index[0]) / scale),
            np.round((stride * t_index[1] + cell_size) / scale),
            np.round((stride * t_index[0] + cell_size) / scale),
            score, offsets
            )
        return bounding_box.T
    def resize_image(self, img, scale):
        """
            resize image and transform dimention to [batchsize, channel, height, width]
        Parameters:
        ----------
            img: numpy array , height x width x channel,input image, channels in BGR order here
            scale: float number, scale factor of resize operation
        Returns:
        -------
            transformed image tensor , 1 x channel x height x width
        """
        height, width, channel = img.shape
        new_height = height
        new_width = width
        img_resized = cv2.resize(img, (new_width, new_height), interpolation=cv2.INTER_LINEAR)

        return img_resized

    def pad(self, bboxes, w, h):
        x1, y1, x2, y2 = [bboxes[:, i] for i in range(4)]
        tmpw = (x2 - x1 + 1).astype(np.int32)
        tmph = (y2 - y1 + 1).astype(np.int32)
        num_boxes = bboxes.shape[0]
        dx, dy, edx, edy = np.zeros((num_boxes, )), np.zeros((num_boxes, )), tmpw.copy() - 1, tmph.copy() - 1
        x, y, ex, ey = x1, y1, x2, y2

        ind = np.where(ex > w - 1)
        edx[ind] = tmpw[ind] + w - 2 - ex[ind]
        ex[ind] = w - 1

        ind = np.where(ey > h - 1)
        edy[ind] = tmph[ind] + h - 2 - ey[ind]
        ey[ind] = h - 1

        ind = np.where(x < 0)
        dx[ind] = 0 - x[ind]
        x[ind] = 0

        ind = np.where(y < 0)
        dy[ind] = 0 - y[ind]
        y[ind] = 0

        return_list = [dy, edy, dx, edx, y, ey, x, ex, tmpw, tmph]
        return_list = [item.astype(np.int32) for item in return_list]

        return return_list

    def detect_pnet(self, im):
        """Get face candidates through pnet

        Parameters:
        ----------
        im: numpy array, input image array

        Returns:
        -------
        boxes: numpy array
            detected boxes before calibration
        boxes_align: numpy array
            boxes after calibration
        """
        h, w, c = im.shape
        net_size = 12
        current_scale = float(net_size) / self.min_face_size   # find initial scale
        im_resized = self.resize_image(im, current_scale)
        current_height, current_width, _ = im_resized.shape

        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        all_boxes = list()
        while min(current_height, current_width) > net_size:
            feed_img = []
            image_tensor = image_tools.convert_image_to_tensor(im_resized)
            feed_img.append(image_tensor)
            feed_img = torch.stack(feed_img)

            if self.pnet_dector.use_cuda is True:
                feed_img = feed_img.to(device)

            cls_map, reg = self.pnet_dector(feed_img)
            cls_map_np = image_tools.convert_chwTensor_to_hwcNumpy(cls_map.cpu())
            reg_np = image_tools.convert_chwTensor_to_hwcNumpy(reg.cpu())

            boxes = self.generate_bounding_box(cls_map_np[0, :, :], reg_np, current_scale, self.threshold[0])

            current_scale *= self.scale_factor
            im_resized = self.resize_image(im, current_scale)
            current_height, current_width, _ = im_resized.shape
            if boxes.size == 0:
                continue
            keep = utils.nms(boxes[:, :5], 0.5, 'union')
            boxes = boxes[keep]
            all_boxes.append(boxes)

        if len(all_boxes) == 0:
            return None, None
        all_boxes = np.vstack(all_boxes)

        keep = utils.nms(all_boxes[:, 0:5], 0.7, 'union')
        all_boxes = all_boxes[keep]

        xx1, yy1, xx2, yy2, scores = [all_boxes[:, i] for i in range(5)]
        offset_x1, offset_y1, offset_x2, offset_y2 = [all_boxes[:, i] for i in range(5, 9)]
        bw = xx2 - xx1 + 1.0
        bh = yy2 - yy1 + 1.0


        boxes = np.vstack([xx1, yy1, xx2, yy2, scores])
        boxes = boxes.T

        x_top = xx1 + offset_x1*bw
        y_top = yy1 + offset_y1*bh
        x_bottom = xx2 + offset_x2*bw
        y_bottom = yy2 + offset_y2*bh

        align_boxes = np.vstack([x_top, y_top, x_bottom, y_bottom, scores])
        align_boxes = align_boxes.T

        return boxes, align_boxes

    def detect_rnet(self, im, dets):
        """Get face candidates using rnet

        Parameters:
        ----------
        im: numpy array
            input image array
        dets: numpy array
            detection results of pnet

        Returns:
        -------
        boxes: numpy array
            detected boxes before calibration
        boxes_align: numpy array
            boxes after calibration
        """
        h, w, c = im.shape
        if dets is None:
            return None, None

        dets = self.square_box(dets)
        dets[:, 0:4] = np.round(dets[:, 0:4])

        [dy, edy, dx, edx, y, ey, x, ex, tmpw, tmph] = self.pad(dets, w, h)
        num_boxes = dets.shape[0]

        cropped_ims_tensors = []

        for i in range(num_boxes):
            try:
                if tmph[i] > 0 and tmpw[i] > 0:
                    tmp = np.zeros((tmph[i], tmpw[i], 3), dtype=np.unit8)
                    tmp[dy[i]:edy[i] + 1, dx[i]: edx[x] + 1, :] = im[y[i]: ey[i] + 1, x[i]: ex[i] + 1, :]
                    crop_im = cv2.resize(tmp, (24, 24))
                    crop_im_tensor = image_tools.convert_image_to_tensor(crop_im)
                    cropped_ims_tensors.append(crop_im_tensor)
            except ValueError as e:
                print(e.message)

        feed_imgs = torch.stack(cropped_ims_tensors)
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        if self.rnet_dector.use_cuda is True:
            feed_imgs = feed_imgs.to(device)

        cls_map, reg = self.rnet_dector(feed_imgs)
        cls_map = cls_map.cpu().data.numpy()
        reg = reg.cpu().data.numpy()

        inds = np.where(cls_map > self.threshold[1])[0]
        if len(inds) == 0:
            return None, None

        boxes = dets[inds]
        cls = cls_map[inds]
        reg = reg[inds]

        keep = utils.nms(boxes, 0.7)
        if len(keep) == 0:
            return None, None

        keep_cls = cls[keep]
        keep_boxes = boxes[keep]
        keep_reg = reg[keep]

        xx1, yy1, xx2, yy2 = [keep_boxes[:, i] for i in range(4)]
        bw = xx2 - xx1
        bh = yy2 - yy1
        boxes = np.vstack([xx1, yy1, xx2, yy2, keep_cls[:, 0]])
        boxes = boxes.T

        off_x1, off_y1, off_x2, off_y2 = [keep_reg[:, i] for i in range(4)]

        x_top = xx1 + off_x1 * bw
        y_top = yy1 + off_y1 * bh
        x_bottom = xx2 + off_x2 * bw
        y_bottom = yy2 + off_y2 *bh
        align_boxes = np.vstack([x_top, y_top, x_bottom, y_bottom, keep_cls[0, 0]])
        align_boxes = align_boxes.T
        return boxes, align_boxes

    def detect_onet(self, im, dets):
        """Get face candidates using onet

               Parameters:
               ----------
               im: numpy array
                   input image array
               dets: numpy array
                   detection results of rnet

               Returns:
               -------
               boxes_align: numpy array
                   boxes after calibration
               landmarks_align: numpy array
                   landmarks after calibration

               """
        if dets is None:
            return None, None

        h, w, c = im.shape

        dets = self.square_box(dets)
        dets[:, 0:4] = np.round(dets[:, 0:4])

        [dy, edy, dx, edx, y, ey, x, ex, tmpw, tmph] = self.pad(dets, w, h)
        num_boxes = dets.shape[0]

        cropped_im_tensor = list()

        for i in range(num_boxes):
            try:
                if tmph > 0 and tmpw > 0:
                    tmp = np.zeros((tmph[i], tmpw[i], 3), dtype=np.uint8)
                    tmp[dy[i]: edy[i] + 1, dx[i]: edx[i] + 1, :] = im[y[i]: ey[i] + 1, x[i]: ex[i] + 1, :]
                    crop_im = cv2.resize(tmp, (48, 48))
                    crop_im_tensor = image_tools.convert_image_to_tensor(crop_im)
                    cropped_im_tensor.append(crop_im_tensor)
            except ValueError as e:
                print(e.message)

        feed_imgs = torch.stack(cropped_im_tensor)
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        if self.onet_dector.use_cuda is True:
            feed_imgs = feed_imgs.to(device)

        cls_map, reg, landmark = self.onet_dector(feed_imgs)
        cls_map = cls_map.cpu().data.numpy()
        reg = reg.cpu().data.numpy()
        landmark = landmark.cpu().data.numpy()

        inds = np.where(cls_map > self.threshold[2])[0]

        if len(inds) == 0:
            return None, None

        boxes = dets[inds]
        cls = cls_map[inds]
        reg = reg[inds]
        land = landmark[inds]

        keep = utils.nms(boxes, 0.7, mode='min')
        if len(keep) == 0:
            return None, None

        keep_boxes = boxes[keep]

        keep_cls = cls[keep]
        keep_reg = reg[keep]
        keep_landmark = land[keep]

        xx1, yy1, xx2, yy2 = [keep_boxes[:, i] for i in range(4)]
        bw, bh = xx2 - xx1, yy2 - yy1

        off_x1, off_y1, off_x2, off_y2 = [keep_reg[:, i] for i in range(4)]
        x_top = xx1 + off_x1*bw
        y_top = yy1 + off_y1*bh
        x_bottom = xx2 + off_x2*bw
        y_bottom = yy2 + off_y2*bh

        landmark_xtop = xx1
        landmark_ytop = yy1

        align_boxes = np.vstack([x_top, y_top, x_bottom, y_bottom, keep_cls[:, 0]])

        align_boxes = align_boxes.T

        landmark = np.vstack([
            landmark_xtop + keep_landmark[:, 0] * bw,
            landmark_ytop + keep_landmark[:, 1] * bh,
            landmark_xtop + keep_landmark[:, 2] * bw,
            landmark_ytop + keep_landmark[:, 3] * bh,
            landmark_xtop + keep_landmark[:, 4] * bw,
            landmark_ytop + keep_landmark[:, 5] * bh,
            landmark_xtop + keep_landmark[:, 6] * bw,
            landmark_ytop + keep_landmark[:, 7] * bh,
            landmark_xtop + keep_landmark[:, 8] * bw,
            landmark_xtop + keep_landmark[:, 9] * bh
        ])

        landmark = landmark.T

        return align_boxes, landmark

    def detect_face(self, img):
        boxes_align = np.array([])
        lamdrk_align = np.array([])
        t = time.time()

        if self.pnet_dector:
            boxes, boxes_align = self.detect_pnet(img)
            if boxes_align is None:
                return np.array([]), np.array([])
            t1 = time.time() - t
            t = time.time()

        if self.rnet_dector:
            boxes, align_boxes = self.detect_rnet(img, boxes_align)
            if boxes_align is None:
                return np.array([]), np.array([])

            t2 = time.time() - t
            t = time.time()

        if self.pnet_dector:
            boxes_align, landmark = self.detect_onet(img, align_boxes)
            if boxes_align is None:
                return np.array([]), np.array([])
            t3 = time.time()

            print("time cost " + '{:.3f}'.format(t1 + t2 + t3) + '  pnet {:.3f}  rnet {:.3f}  onet {:.3f}'.format(t1, t2,
                                                                                                                t3))

        return boxes_align, landmark




