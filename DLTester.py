from edgegan.utils.data.dataset import dataset
from torch.utils.data import DataLoader
import os

if __name__ == '__main__':

    cwd = os.getcwd()
    print("cwd", cwd)
    ds = dataset(dataroot="../../../images", dataset="data")
    dl = DataLoader(dataset=ds, batch_size=16, shuffle=True)

    for i_batch, data_batch in enumerate(dl):
        img, z = data_batch
        print(img.shape)
        print(z.shape)
        break
