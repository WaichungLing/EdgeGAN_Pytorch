import os
import torch
import glob
from pathlib import Path
from torch.utils.data.dataset import Dataset
from PIL import Image
from torchvision import transforms
import numpy as np


def extension_match_recursive(root, exts):
    result = []
    for ext in exts:
        paths = [str(p) for p in Path(root).rglob(ext)]
        result.extend(paths)
    return result


def one_hot(labels: torch.Tensor, num_classes: int, dtype: torch.dtype = torch.float, dim: int = 1) -> torch.Tensor:
    # if `dim` is bigger, add singleton dim at the end
    if labels.ndim < dim + 1:
        shape = list(labels.shape) + [1] * (dim + 1 - len(labels.shape))
        labels = torch.reshape(labels, shape)

    sh = list(labels.shape)

    if sh[dim] != 1:
        raise AssertionError("labels should have a channel with length equal to one.")

    sh[dim] = num_classes

    o = torch.zeros(size=sh, dtype=dtype, device=labels.device)
    labels = o.scatter_(dim=dim, index=labels.long(), value=1)

    return labels


class dataset(Dataset):
    def __init__(self, dataroot, dataset, zdim=100, num_classes=5, phase='train'):
        assert phase in ['train', 'test']
        self.num_classes = num_classes
        self.phase = phase
        self.zdim = zdim
        self.data = []
        if num_classes is not None:
            for i in range(num_classes):
                i_dir = os.path.join(dataroot, dataset, phase, str(i), '*.png')
                for filename in glob.glob(i_dir):
                    self.data.append(filename)
        else:
            data_path = os.path.join(dataroot, dataset, phase, '*.png')
            for filename in glob.glob(data_path):
                self.data.append(filename)

        if len(self.data) == 0:
            raise Exception("[!] No data found in '" + data_path + "'")
        self.size = len(self.data)
        print("Dataset size: ", self.size)

    def shuffle(self):
        np.random.shuffle(self.data)

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        filenames = self.data[idx]
        image = Image.open(filenames)

        transform = transforms.Compose([
            transforms.ToTensor()
        ])

        image = transform(image)

        if self.phase == 'train':
            z = torch.normal(mean=0.0, std=1.0, size=(self.zdim,))

            if self.num_classes is not None:
                def get_class(filePath):
                    end = filePath.rfind("/")
                    start = filePath.rfind("/", 0, end)
                    return int(filePath[start + 1:end])

                onehot = one_hot(torch.tensor(get_class(filenames)), self.num_classes)[0]

                z = np.concatenate((z, onehot))

        return (image, z) if self.phase == 'train' else image
