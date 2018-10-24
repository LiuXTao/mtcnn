import sys
import os
import datetime
import torch
import torch.optim as optim
import tools.config as config
from tools.image_reader import TrainImageReader
from nets.models import ONet
from nets.nets_utils import LossFn, compute_accuracy
import tools.image_tools as image_tools
from tools.imagedb import ImageDB


def train_onet(anno_file, model_store, end_epoch, frequent, base_lr, batch_size, use_cuda=False):
    if not os.path.exists(model_store):
        os.makedirs(model_store)

    onet = ONet(is_train=True, use_cuda=use_cuda)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    if use_cuda is True:
        onet = onet.to(device)

    loss_function = LossFn()
    optimizer = optim.Adam(onet.parameters(), lr=base_lr)
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[10, 25, 40], gamma=0.1)

    # 加载训练图片
    imagedb = ImageDB(anno_file)
    gt_imdb = imagedb.load_imdb()
    gt_imdb = imagedb.append_flipped_images(gt_imdb)

    train_data = TrainImageReader(gt_imdb, 48, batch_size, shuffle=True)

    # training rnet
    onet.train()

    for cur_epoch in range(end_epoch):
        scheduler.step()
        train_data.reset()

        for batch_idx, (image, (gt_label, gt_boxes, gt_landmark)) in enumerate(train_data):
            im_tensor = [image_tools.convert_image_to_tensor(image[i, :, :, :]) for i in range(image.shape[0])]
            in_tensor = torch.stack(im_tensor)

            gt_label = torch.from_numpy(gt_label)
            gt_boxes = torch.from_numpy(gt_boxes)

            if use_cuda is True:
                gt_label = gt_label.to(device)
                gt_boxes = gt_boxes.to(device)
                im_tensor = im_tensor.to(device)

            cls_pred, box_offset_pred = rnet(im_tensor)
            cls_loss = loss_function.cls_loss(cls_pred, gt_label)
            boxes_loss = loss_function.box_loss(gt_label, gt_boxes, box_offset_pred)
            total_loss = cls_loss * 1.0 + boxes_loss * 0.5

            if batch_idx % frequent == 0:
                accuracy = compute_accuracy(cls_pred, gt_label)

                print(
                    "[%s, Epoch: %d, Step: %d] accuracy: %.6f, all_loss: %.6f, cls_loss: %.6f, bbox_reg_loss: %.6f, lr: %.6f" %
                    (datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'), cur_epoch + 1, batch_idx,
                     accuracy.data.tolist(),
                     total_loss.data.tolist(), cls_loss.data.tolist(), boxes_loss.data.tolist(),
                     scheduler.get_lr()[0]))
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()

        if batch_idx % 10 == 0:
            torch.save(onet.state_dict(), os.path.join(model_store, 'onet_model_epoch_%d.pt' % (batch_idx + 1)))
    torch.save(onet.state_dict(), os.path.join(model_store, 'onet_model_final.pt'))


if __name__ == '__main__':
    anno_file = os.path.join(config.ANNO_STORE_DIR, config.PNET_TRAIN_IMGLIST_FILENAME)
    model_store = config.MODLE_STORE_DIR
    end_epoch = config.END_EPOCH
    frequent = 200
    base_lr = config.TRAIN_LR
    batch_size = config.TRAIN_BATCH_SIZE

    train_onet(anno_file, model_store, end_epoch, frequent, base_lr, batch_size)
