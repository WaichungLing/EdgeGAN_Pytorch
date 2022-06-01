import os
import glob
import scipy.misc
from edgegan.utils import makedirs

gt_dir = 'Data/GT/train'
out_dir = 'Animals/GT'


def convert(classes):
    idx = 0
    for c in classes:
        c_dir = os.path.join(gt_dir, str(c) + '/*.png')
        out_path = os.path.join(out_dir, str(idx))
        makedirs(out_path)
        idx += 1

        for filename in glob.glob(c_dir):
            f = filename[filename.rfind('/') + 1:]
            img = scipy.misc.imread(filename, mode='RGB')
            assert img.shape[-1] == 3
            scipy.misc.imsave(os.path.join(out_path, f), img)


if __name__ == '__main__':
    convert([17, 18, 20, 24, 25])
