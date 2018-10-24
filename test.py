import sys
import os
import cv2

import tools.visual_tools as vision
from tools.detect_utils import create_mtcnn_net
from train.MtcnnDetector import MtcnnDector

if __name__ == '__main__':
    pnet, rnet, onet = create_mtcnn_net(pnet_path='./model_store/pnet_model_final.pt',
                                        rnet_path='./model_store/rnet_model_final.pt',
                                        onet_path='./model_store/onet_model_final.pt', use_cuda=False)

    mtcnn_detector = MtcnnDector(pnet=pnet, rnet=rnet, onet=onet, min_face_size=24)

    img = cv2.imread('./test.jpg')
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    bboxes, landmark = mtcnn_detector.detect_face(img)
    vision.visual_face(img, bboxes, landmark)

