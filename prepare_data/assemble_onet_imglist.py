import sys
import os
import tools.config as config
import prepare_data.assemble as assemble

if __name__ == '__main__':
    anno_file = []
    onet_positive_file = os.path.join(config.ANNO_STORE_DIR, config.ONET_POSTIVE_ANNO_FILENAME)
    onet_negative_file = os.path.join(config.ANNO_STORE_DIR, config.ONET_NEGATIVE_ANNO_FILENAME)
    onet_part_file = os.path.join(config.ANNO_STORE_DIR, config.ONET_PART_ANNO_FILENAME)

    anno_file.append(onet_positive_file)
    anno_file.append(onet_part_file)
    anno_file.append(onet_negative_file)

    image_list = os.path.join(config.ANNO_STORE_DIR, config.PNET_TRAIN_IMGLIST_FILENAME)

    chose_count = assemble.assemble_data(image_list, anno_file)
    print("RNet train annotation result file path:%s, total num of imgs: %d" % (image_list, chose_count))


