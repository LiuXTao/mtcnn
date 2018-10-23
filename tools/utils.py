import numpy as np

def IoU(box, boxes):
    """Compute IoU between detect box and gt boxes

        Parameters:
        ----------
        box: numpy array , shape (5, ): x1, y1, x2, y2, score
            input box
        boxes: numpy array, shape (n, 4): x1, y1, x2, y2
            input ground truth boxes

        Returns:
        -------
        ovr: numpy.array, shape (n, )
            IoU
    """
    x1, y1, x2, y2, score = [box[i] for i in range(5)]
    w, h = x2 - x1, y2 - y1
    box_area = w * h

    xs1, ys1, xs2, ys2, scores = [boxes[:, i] for i in range(5)]
    area = (ys2 - ys1) * (xs2 - xs1)

    xx1 = np.maximum(x1, xs1)
    xx2 = np.maximum(x2, xs2)
    yy1 = np.maximum(y1, ys1)
    yy2 = np.maximum(y2, ys2)

    width = np.maximum(0, xx2 - xx1)
    height = np.maximum(0, yy2 - yy1)

    inter = width * height
    ovr = np.true_divide(inter, (box_area + area - inter))

    return ovr

def convert_to_square(box):
    ''' Convert bbox to a square which it can include the bbox
        Parameters:
            bbox: numpy array, shape n x 5

        returns:
            square box
    '''

    x1, y1, x2, y2 = [box[:, i] for i in range(4)]
    square = box.copy()
    w, h = x2 - x1, y2 - y1
    max_side = np.maximum(h, w)
    square[:, 0] = x1 + w*0.5 - max_side*0.5
    square[:, 1] = y1 + w*0.5 - max_side*0.5
    square[:, 2] = square[:, 0] + max_side
    square[:, 3] = square[:, 1] +max_side

    return square


def nms(bboxes, overlap_threshold, mode='union'):
    ''' greedily select bboxes with high confidence,if an box overlap with the highest score box > thres, rule it out

        params:
            dets: [[x1, y1, x2, y2, score]]
            thresh: retain overlap <= thresh
        return:
            indexes to keep
    '''
    if len(bboxes) == 0:
        return []
    picked_list = []
    x1, y1, x2, y2, scores = [bboxes[:, i] for i in range(5)]

    area = (x2 - x1) * (y2 - y1)
    ids = scores.argsort()[::-1]

    while ids.size >0:
        biggest = ids[0]
        picked_list.append(biggest)

        xx1 = np.maximum(x1[biggest], x1[ids[1:]])
        yy1 = np.maximum(y1[biggest], y1[ids[1:]])
        xx2 = np.maximum(x2[biggest], x2[ids[1:]])
        yy2 = np.maximum(y2[biggest], y2[ids[1:]])

        w = np.maximum(0.0, xx2 - xx1)
        h = np.maximum(0.0, yy2 - yy1)

        inter_area = w *h

        if mode == 'min':
            overlap = inter_area / np.minimum(area[biggest], area[ids[1:]])
        elif mode == 'union':
            overlap = inter_area / (area[biggest] + area[ids[1:]] - inter_area)

        inds = np.where(overlap <= overlap_threshold)[0]
        ids = ids[inds + 1]
    return picked_list