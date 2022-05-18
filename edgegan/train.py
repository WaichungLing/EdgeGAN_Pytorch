import os
import json
import numpy as np
from argparse import ArgumentParser

import torch

from edgegan.utils import makedirs
from edgegan.utils.data import Dataset


def make_outputs_dir(args):
    path = os.path.join(args.outputsroot, args.name)

    makedirs(args.outputsroot)
    makedirs(os.path.join(path, 'checkpoints'))
    makedirs(os.path.join(path, 'logs'))


def save_flags(args):
    path = os.path.join(args.outputsroot, args.name)
    with open(os.path.join(path, 'flags.json'), 'w') as f:
        json.dump(args.__dict__, f, indent=4)


def main(args):
    phase = 'train'

    if args.input_width is None:
        args.input_width = args.input_height
    if args.output_width is None:
        args.output_width = args.output_height

    if not args.multiclasses:
        args.num_classes = None

    dataset_config = {
        'input_height': args.input_height,
        'input_width': args.input_width,
        'output_height': args.output_height,
        'output_width': args.output_width,
        'crop': args.crop,
        'grayscale': False,
        'z_dim': args.z_dim,
    }

    #============================#
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    #============================#

    dataset = Dataset(
        args.dataroot, args.dataset,
        args.train_size, args.batch_size,
        dataset_config, args.num_classes, phase)
    edgegan_model = EdgeGAN(args, dataset, device, z_dim=args.z_dim)
    edgegan_model.train()


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--gpu', type=str, default='0', help='GPU id')
    parser.add_argument('--name', type=str, default='edgegan', help='Folder for all outputs')
    parser.add_argument('--outputsroot', type=str, default='outputs', help='Outputs root')
    parser.add_argument('--dataset', type=str, default='data', help='The dataset that contains train/test directory')
    parser.add_argument('--dataroot', type=str, default='images', help='The directory contains --dataset')
    parser.add_argument('--input_fname_pattern', type=str, default='*png', help='Glob pattern of input image')
    parser.add_argument('--checkpoint_dir', type=str, default=None, help='Checkpoint directory')
    parser.add_argument('--log_dir', type=str, default=None, help='Log directory')
    parser.add_argument('--save_checkpoint_frequency', type=int, default=500, help='Frequency for saving checkpoint')

    parser.add_argument('--epoch', type=int, default=100, help='Epoch to train')
    parser.add_argument('--learning_rate', type=float, default=0.0002, help='Training learning rate')
    parser.add_argument('--train_size', type=float, default=np.inf, help='Max number of train samples')
    parser.add_argument('--batch_size', type=int, default=64, help='Training batch size')
    parser.add_argument('--stage1_zl_loss', type=float, default=10.0, help='Weight of z l1 loss')
    parser.add_argument('--input_height', type=int, default=64, help='The height of input image')
    parser.add_argument('--input_width', type=int, default=128, help='The width of input image')
    parser.add_argument('--output_height', type=int, default=64, help='The height of output image')
    parser.add_argument('--output_width', type=int, default=128, help='The width of output image')
    parser.add_argument('--crop', type=bool, default=False, help='Cropping input or not')  # ?

    parser.add_argument('--if_resnet_e', type=bool, default=True, help='If use resnet for E')
    parser.add_argument('--if_resnet_g', type=bool, default=False, help='If use resnet for G')
    parser.add_argument('--if_resnet_d', type=bool, default=False, help='If use resnet for D')
    parser.add_argument('--lambda_gp', type=float, default=10.0, help='')  # ?
    parser.add_argument('--E_norm', type=str, default='instance', help='normalization options:[instance, batch, norm]')
    parser.add_argument('--G_norm', type=str, default='instance', help='normalization options:[instance, batch, norm]')
    parser.add_argument('--D_norm', type=str, default='instance', help='normalization options:[instance, batch, norm]')

    parser.add_argument('--use_image_discriminator', type=bool, default=True, help='True for using patch '
                                                                                   'discriminator, modify the size of'
                                                                                   ' input of discriminator')
    parser.add_argument('--image_dis_size', type=int, default=128, help='The size of input for image discriminator')
    parser.add_argument('--use_edge_discriminator', type=bool, default=True, help='True for using patch '
                                                                                  'discriminator, modify the size of '
                                                                                  'input of discriminator, '
                                                                                  'user for edge discriminator when '
                                                                                  'G_num == 2')
    parser.add_argument('--edge_dis_size', type=int, default=128, help='The size of input for edge discriminator')
    parser.add_argument('--joint_dweight', type=float, default=1.0, help='weight of joint discriminative loss')
    parser.add_argument('--image_dweight', type=float, default=1.0, help='weight of image discriminative loss, '
                                                                         'is ineffective when use_image_discriminator'
                                                                         ' is false')
    parser.add_argument('--edge_dweight', type=float, default=1.0, help='weight of edge discriminative loss, '
                                                                        'is ineffective when use_edge_discriminator '
                                                                        'is false')
    parser.add_argument('--z_dim', type=int, default=100, help='dimension of random vector z')

    # Multi-class
    parser.add_argument('--multiclasses', type=bool, default=True, help='If use focal loss')
    parser.add_argument('--num_classes', type=int, default=5, help='Number of classes')
    parser.add_argument('--SPECTRAL_NORM_UPDATE_OPS', type=str, default='spectral_norm_update_ops')  # ?
    parser.add_argument('--SPECTRAL_NORM_UPDATE_OPS', type=str, default='spectral_norm_update_ops')  # ?

    args = parser.parse_args()

    main(args)
