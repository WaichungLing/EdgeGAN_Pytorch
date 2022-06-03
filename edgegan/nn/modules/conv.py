import torch
import torch.nn as nn

from .upsampling import upsample
from .pooling import mean_pool
from .normalization import spectral_normed_weight, ADN


def conv2d2(inputs, num_outputs, kernel_size, sn, stride=1):
    channel_axis = 1
    input_dim = inputs.shape[channel_axis]

    m = nn.Conv2d(input_dim, num_outputs, kernel_size, stride=stride, padding='same')
    torch.nn.init.xavier_uniform_(m.weight)

    if sn:
        m.weight = nn.Parameter(spectral_normed_weight(m.weight, num_iters=1))

    if inputs.is_cuda:
        m = m.cuda()
        conv_out = m(inputs)
    else:
        conv_out = m(inputs)

    return conv_out


class MRU_Cell(nn.Module):
    def __init__(self, filter_depth, sn, stride, dilate=1, norm_input=True,
                 deconv=False):
        super(MRU_Cell, self).__init__()
        self.filter_depth = filter_depth
        self.sn = sn
        self.stride = stride
        self.dilate_rate = dilate
        self.norm_input = norm_input
        self.deconv = deconv
        self.adn = ADN()

    def forward(self, inp, ht):

        channel_index = 1
        reduce_dim = [2, 3]
        hidden_depth = ht.shape[channel_index]

        if self.deconv:
            if self.stride == 2:
                ht = upsample(ht)
            elif self.stride != 1:
                raise NotImplementedError

        ht_orig = ht.clone()

        if self.norm_input:
            full_inp = torch.cat((self.adn(ht), inp), channel_index)
        else:
            full_inp = torch.cat((ht, inp), channel_index)

        # update gate
        rg = conv2d2(full_inp, hidden_depth, 3, sn=self.sn, stride=1)
        rg = self.adn(rg)
        rg = (rg - torch.amin(rg, dim=reduce_dim, keepdim=True)) / (
                torch.amax(rg, dim=reduce_dim, keepdim=True) - torch.amin(rg, dim=reduce_dim, keepdim=True))

        # Input Image conv
        img_new = conv2d2(inp, hidden_depth, 3, sn=self.sn, stride=1)

        ht_plus = ht + rg * img_new
        ht_new_in = self.adn(ht_plus)

        # new hidden state
        h_new = conv2d2(ht_new_in, self.filter_depth, 3, sn=self.sn, stride=1)
        h_new = self.adn(h_new)
        h_new = conv2d2(h_new, self.filter_depth, 3, sn=self.sn, stride=1)

        if ht.shape[channel_index] != self.filter_depth:
            ht_orig = conv2d2(ht_orig, self.filter_depth, 1, sn=self.sn, stride=1)
        ht_new = ht_orig + h_new

        if not self.deconv:
            if self.stride == 2:
                ht_new = mean_pool(ht_new)
            elif self.stride != 1:
                raise NotImplementedError

        return ht_new


class MRU(nn.Module):
    def __init__(self, filter_depth, sn, stride=2, dilate_rate=1,
                 num_blocks=5, last_unit=False):
        super(MRU, self).__init__()
        self.filter_depth = filter_depth
        self.sn = sn
        self.stride = stride
        self.dilate_rate = dilate_rate
        self.num_blocks = num_blocks
        self.last_unit = last_unit

        self.cell = MRU_Cell(self.filter_depth, self.sn, self.stride,
                             dilate=self.dilate_rate, deconv=False)

        self.adn = ADN()

    def forward(self, x, ht):  # input = (64,3,64,64) hts[0] = (64, depth, outh, outw)
        if self.dilate_rate != 1:
            stride = 1

        hts_new = []
        inp = x

        ht_new = self.cell(inp, ht[0])
        hts_new.append(ht_new)
        inp = ht_new

        if self.last_unit:
            hts_new[-1] = self.adn(hts_new[-1])

        return hts_new
