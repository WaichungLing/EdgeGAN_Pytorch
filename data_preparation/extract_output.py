#
# Extract EdgeGAN input/output into a directory for FID comparison
# Result: a directory containing 3*64*64 images
#
from argparse import ArgumentParser
import glob
import os

import numpy as np
import scipy.misc

def extract(args):
    print(os.path.join("..", args.inpath))
    dataroot = os.path.join("..", args.inpath)
    cats = [name for name in os.listdir(dataroot) if os.path.isdir(os.path.join(dataroot, name))]
    for cat in cats:
        c_dir = os.path.join(dataroot, cat)+"/*.png"

        # 0-63 64-127 128-191 192-255
        for filename in glob.glob(c_dir):
            f = filename[filename.rfind('/') + 1:]
            if args.original:
                img = scipy.misc.imread(filename, mode='RGB')[:, 64:127, :]
            else:
                img = scipy.misc.imread(filename, mode='RGB')[:, 192:, :]

            scipy.misc.imsave(os.path.join(args.outpath, f), img)

#
# --inpath: path to a directory, which contains several directories that indicate a category
# --outpath: output path, can be arbitrary
# --original: if True: crop the right-hand side of training set; if false, crop the last patch of test output
if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument("--inpath", type=str, default="")
    parser.add_argument("--outpath", type=str, default="out")
    parser.add_argument("--original", type=bool, default=False)
    args = parser.parse_args()
    os.system('mkdir -p {}'.format(args.outpath))
    extract(args)