import torch
from edgegan.models.edgegan import EdgeGAN
from edgegan.utils.data.dataset import dataset
from torch.utils.data import DataLoader

if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"running on {device}")

    train_data = dataset(dataroot="./images", dataset="data")
    train_dl = DataLoader(dataset=train_data, batch_size=64, shuffle=True)

    edgegan = EdgeGAN().to(device)
    edgegan.train(train_dl, device=device)


