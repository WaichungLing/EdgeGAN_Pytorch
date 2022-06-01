import os
import json
import numpy as np
from argparse import ArgumentParser

import torch

from edgegan.utils import makedirs
from edgegan.utils.data import Dataset


def subdirs(root):
    return [name for name in os.listdir(root)
            if os.path.isdir(os.path.join(root, name))]


def make_outputs_dir(args, test_output_dir):
    makedirs(os.path.join(test_output_dir, args.dataset))
    for path in subdirs(os.path.join(args.dataroot, args.dataset, 'test')):
        makedirs(os.path.join(test_output_dir, args.dataset, path))


def main(args):
    phase = 'test'

    if args.input_width is None:
        args.input_width = args.input_height
    if args.output_width is None:
        args.output_width = args.output_height

    args.batch_size = 1

    path = os.path.join(args.outputsroot, args.name)
    test_output_dir = os.path.join(path, 'test_output')
    make_outputs_dir(args, test_output_dir)

    dataset_config = {
        'input_height': args.input_height,
        'input_width': args.input_width,
        'output_height': args.output_height,
        'output_width': args.output_width,
        'crop': args.crop,
        'grayscale': False,
    }

    dataset = Dataset(
        args.dataroot, args.dataset,
        args.train_size, 1,
        dataset_config, None, phase
    )

    # ============================#
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # ============================#

    edgegan = EdgeGAN(args, dataset, device)
    edgegan.test()


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--gpu', type=str, default='0', help='GPU id')
    parser.add_argument('--name', type=str, default='edgegan', help='Folder for all outputs')
    parser.add_argument('--outputsroot', type=str, default='outputs', help='Outputs root')
    parser.add_argument('--epoch', type=int, default=100, help='Epoch to train [25]')
    parser.add_argument('--learning_rate', type=float, default=0.0002, help='Learning rate')
    parser.add_argument('--train_size', type=float, default=np.inf, help='The size of train images [np.inf]')
    parser.add_argument('--batch_size', type=int, default=1, help='Batch size')
    parser.add_argument('--input_height', type=int, default=64, help='The size of image to use')
    parser.add_argument('--input_width', type=int, default=128, help='The size of the image to use')
    parser.add_argument('--output_height', type=int, default=64, help='The size of the output images to produce')
    parser.add_argument('--output_width', type=int, default=128, help='The size of the output images to produce')

    parser.add_argument('--dataset', type=str, default='data')
    parser.add_argument('--input_fname_pattern', type=str, default='*png', help='Glob pattern of filename of input '
                                                                                'images')
    parser.add_argument('--checkpoint_dir', type=str, default=None)
    parser.add_argument('--logdir', type=str, default=None)
    parser.add_argument('--dataroot', type=str, default='./images', help='Root directory of dataset')
    parser.add_argument('--test_output_dir', type=str, default='test_output')
    parser.add_argument('--crop', type=bool, default=False)

    parser.add_argument('--output_combination', type=str, default='full', help='The combination of output image: '
                                                                               'full(input+output), inputL_outputR('
                                                                               'the left of input combine the right '
                                                                               'of output),outputL_inputR, outputR')

    parser.add_argument('--multiclasses', type=bool, default=True, help='If use multiclass model and focal loss')
    parser.add_argument('--num_classes', type=int, default=5, help='Num of classes')
    parser.add_argument('--type', type=str, default='gpwgan', help='an type: [dcgan | wgan | gpwgan]')
    parser.add_argument('--optim', type=str, default='rmsprop', help='optimizer type: [adam | rmsprop]')
    parser.add_argument('--model', type=str, default='old', help='which base model(G and D): [old | new]')
    parser.add_argument('--if_resnet_e', type=bool, default=True, help='If use resnet for E')
    parser.add_argument('--if_resnet_g', type=bool, default=False, help='If use resnet for G')
    parser.add_argument('--if_resnet_d', type=bool, default=False, help='If use resnet for D')
    parser.add_argument('--E_norm', type=str, default='instance', help='normalization options:[instance, batch, norm]')
    parser.add_argument('--G_norm', type=str, default='instance', help='normalization options:[instance, batch, norm]')
    parser.add_argument('--D_norm', type=str, default='instance', help='normalization options:[instance, batch, norm]')

    args = parser.parse_args()
    main(args)
