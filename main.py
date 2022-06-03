from edgegan.models.edgegan import EdgeGAN
from edgegan.utils import makedirs
from edgegan.utils.data.dataset import dataset
from torch.utils.data import DataLoader

if __name__ == '__main__':
    train_data = dataset(dataroot="./images", dataset="data")
    train_dl = DataLoader(dataset=train_data, batch_size=64, shuffle=True)

    checkpoint_dir = "./output"
    makedirs(checkpoint_dir)

    edgegan = EdgeGAN()
    edgegan.train(train_dl, checkpoint_dir)


