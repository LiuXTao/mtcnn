import matplotlib.pyplot as plt

def visual_face(img, bounding_boxes, landmarks=None):
    figure = plt.figure()
    plt.imshow(img)
    figure.suptitle('Face Detector', fontsize=12, color='r')

    for b in bounding_boxes:
        rect = plt.Rectangle((b[0], b[1]),
                             b[2] - b[0],
                             b[3] - b[1],
                             fill=False, edgecolor='yellow', linewidth=0.9
                             )
        plt.gca().add_patch(rect)

    if landmarks is not None:
        for one in landmarks:
            one = one.reshape((5, 2))
            for i in range(5):
                plt.scatter(one[i, 0], one[i, 1], c='red', linewidths=1, marker='x', s=5)

    plt.show()

def visual_two(img, bounding_box1, bounding_box2, thres=0.9):
    figure = plt.figure()
    plt.subplot(121)
    plt.imshow(img)
    figure.suptitle('Face Detector', fontsize=12, color='r')

    for b in bounding_box1:
        score = b[4]
        landmarks = b[5:]
        if score > thres:
            rect = plt.Rectangle((b[0], b[1]),
                                 b[2] - b[0],
                                 b[3] - b[1],
                                 fill=False, edgecolor='red', linewidth=0.7)
            plt.gca().add_patch(rect)
            landmarks = landmarks.reshape((5,2))
            for i in range(5):
                plt.scatter(landmarks[i, 0], landmarks[i, 1], c='yellow', linewidths=1, marker='x', s=20)

    plt.subplot(122)
    plt.imshow(img)
    for b in bounding_box2:
        score = b[4]
        landmarks = b[5:]
        if score > thres:
            rect = plt.Rectangle((b[0], b[1]),
                                 b[2] - b[0],
                                 b[3] - b[1],
                                 fill=False, edgecolor='red', linewidth=0.7)
            plt.gca().add_patch(rect)
            landmarks = landmarks.reshape((5,2))
            for i in range(5):
                plt.scatter(landmarks[i, 0], landmarks[i, 1], c='yellow', linewidths=1, marker='x', s=20)
    plt.show()