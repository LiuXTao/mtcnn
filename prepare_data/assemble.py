import os
import numpy.random as npr
import numpy as np

def assemble_data(output_file, anno_file_list=[]):
    # assemble the annotations to one file
    size = 12
    if len(anno_file_list) == 0:
        return 0

    if os.path.exists(output_file):
        os.remove(output_file)

    chose_count = 0
    for anno_file in anno_file_list:
        with open(anno_file, 'r') as f:
            annos = f.readlines()

        base_num = 250000
        if len(annos) > base_num*3:
            idx_keep = npr.choice(len(annos), size=base_num*3, replace=True)
        elif len(annos) > 100000:
            idx_keep = npr.choice(len(annos), size=len(annos), replace=True)
        else:
            idx_keep = np.arange(len(annos))
            npr.shuffle(idx_keep)

        with open(output_file, 'a+') as f:
            for idx in idx_keep:
                f.write(annos[idx])
                chose_count += 1
    return chose_count