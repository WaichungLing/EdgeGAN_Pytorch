def mean_pool(x, data_format):
    assert data_format == 'NCHW'
    output = (x[:, :, ::2, ::2] + x[:, :, 1::2, ::2] + x[:, :, ::2, 1::2] + x[:, :, 1::2, 1::2]) / 4.
    return output
