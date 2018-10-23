import sys
import os
import tools.config as config
import prepare_data.assemble as assemble

if __name__ == '__main__':
    anno_list = []

    pnet_positive_file = os.path.join(config.ANNO_STORE_DIR, config.PNET_POSTIVE_ANNO_FILENAME)
    pnet_negative_file = os.path.join(config.ANNO_STORE_DIR, config.PNET_NEGATIVE_ANNO_FILENAME)
    pnet_part_file = os.path.join(config.ANNO_STORE_DIR, config.PNET_PART_ANNO_FILENAME)

    for i in [pnet_positive_file, pnet_negative_file, pnet_part_file]:
        anno_list.append(i)

    imglist_file = os.path.join(config.ANNO_STORE_DIR, config.PNET_TRAIN_IMGLIST_FILENAME)
    chose_count = assemble.assemble_data(imglist_file, anno_list)
    print("PNet train annotation result file path:%s, total num of imgs: %d" % (imglist_file, chose_count))



