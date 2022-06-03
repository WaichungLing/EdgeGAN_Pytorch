import torch
from torch import nn
import config
from edgegan.nn.modules import MRU, FullyConnected, mean_pool, conv2d2


class Classifier(nn.Module):
    def __init__(self, sn=True, num_classes=5):
        super(Classifier, self).__init__()
        self.sn = sn
        self.num_classes = num_classes
        self.size = 64
        self.num_blocks = 1

        self.mru1 = MRU(self.size * 2, sn=self.sn, stride=2, dilate_rate=1, num_blocks=self.num_blocks,
                        last_unit=False)
        self.mru2 = MRU(self.size * 4, sn=self.sn, stride=2, dilate_rate=1, num_blocks=self.num_blocks,
                        last_unit=False)
        self.mru3 = MRU(self.size * 8, sn=self.sn, stride=2, dilate_rate=1, num_blocks=self.num_blocks,
                        last_unit=False)
        self.mru4 = MRU(self.size * 12, sn=self.sn, stride=2, dilate_rate=1, num_blocks=self.num_blocks,
                        last_unit=False)
        self.mlp = FullyConnected(self.sn, self.size * 12, self.num_classes)

    def forward(self, x):
        channel_axis = 1

        if type(x) is list:
            x = x[-1]

        x_list = []
        resized_ = x
        x_list.append(resized_)
        for i in range(5):
            resized_ = mean_pool(resized_)
            x_list.append(resized_)
        x_list = x_list[::-1]

        output_dim = 1

        h0 = conv2d2(x_list[-1], 8, kernel_size=7, sn=self.sn, stride=1)
        hts_0 = [h0]

        hts_1 = self.mru1(x_list[-1], hts_0)
        hts_2 = self.mru2(x_list[-2], hts_1)
        hts_3 = self.mru3(x_list[-3], hts_2)
        hts_4 = self.mru4(x_list[-4], hts_3)

        img = hts_4[-1]
        img_shape = img.shape

        disc = conv2d2(img, output_dim, kernel_size=1, sn=self.sn, stride=1)
        img = torch.mean(img, dim=(2, 3))
        logits = self.mlp(img)

        return disc, torch.sigmoid(logits), logits