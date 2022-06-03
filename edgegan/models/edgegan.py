import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.transforms import Resize
from edgegan.nn.functional import *
from .discriminator import Discriminator
from .encoder import Encoder
from .generator import Generator
from .classifier import Classifier

class EdgeGAN(nn.Module):
    def __init__(self, image_d_size=128, edge_d_size=128, num_classes=5, z_dim=100): 
        super(EdgeGAN, self).__init__()
        self.edge_generator = Generator(64, 64)
        self.image_generator = Generator(64, 64)
        self.joint_discriminator = Discriminator(64, 128)
        self.edge_discriminator = Discriminator(edge_d_size, edge_d_size)
        self.image_discriminator = Discriminator(image_d_size, image_d_size)
        self.encoder = Encoder()
        self.classifier = Classifier()
        self.image_d_size = image_d_size
        self.edge_d_size = edge_d_size
        self.num_classes = num_classes
        self.z_dim = z_dim

    def forward(self, x, z):
        self.x = x
        self.z = z
        self.edge_g_output = self.edge_generator(z)
        self.image_g_output = self.image_generator(z)
        self.joint_g_output = torch.cat((self.edge_g_output, self.image_g_output), 3)

        _, self.true_joint_d_output = self.joint_discriminator(x)
        _, self.fake_joint_d_output = self.joint_discriminator(self.joint_g_output)

        crop = int((x.size()[-1])/2)

        images = x[:,:,:,crop:]
        self.resized_images = Resize(self.image_d_size)(images)
        _, self.true_image_d_output = self.image_discriminator(self.resized_images)
        self.resized_image_g_output = Resize(self.image_d_size)(self.image_g_output)
        _, self.fake_image_d_output = self.image_discriminator(self.resized_image_g_output)

        edges = x[:,:,:,:crop]
        self.resized_edges = Resize(self.edge_d_size)(edges)
        _, self.true_edge_d_output = self.edge_discriminator(self.resized_edges)
        self.resized_edge_g_output = Resize(self.edge_d_size)(self.edge_g_output)
        _, self.fake_edge_d_output = self.edge_discriminator(self.resized_edge_g_output)

        _, _, self.true_image_c_output = self.classifier(images)
        _, _, self.fake_image_c_output = self.classifier(self.image_g_output)

        self.e_output = self.encoder(self.edge_g_output)

        return self.joint_g_output

    def compute_loss(self):
        self.joint_dis_dloss = discriminator_ganloss(self.true_joint_d_output, self.fake_joint_d_output)
            #+ penalty(self.joint_g_output, self.x, self.joint_discriminator, self.batch_size)
        self.joint_dis_gloss = generator_ganloss(self.fake_joint_d_output)

        self.image_dis_dloss = discriminator_ganloss(self.true_image_d_output, self.fake_image_d_output)
            #+ penalty(self.resized_image_g_output, self.resized_images, self.image_discriminator, self.batch_size)
        self.image_dis_gloss = generator_ganloss(self.fake_image_d_output)

        self.edge_dis_dloss = discriminator_ganloss(self.fake_edge_d_output, self.true_edge_d_output)
            #+ penalty(self.resized_edge_g_output, self.resized_edges, self.edge_discriminator, self.batch_size)
        self.edge_dis_gloss = generator_ganloss(self.fake_edge_d_output)

        self.edge_gloss = self.joint_dis_gloss + self.edge_dis_gloss
        self.image_gloss = self.joint_dis_gloss + self.image_dis_gloss

        self.loss_g_ac, self.loss_d_ac = get_acgan_loss_focal(
            self.true_image_c_output, 
            self.z[:,-1].long(),
            self.fake_image_c_output, 
            self.z[:,-1].long(),
            self.num_classes
        )

        self.zl_loss = l1loss(self.z[:,:self.z_dim], self.e_output, weight=10.0)

        return (self.joint_dis_dloss + self.image_dis_dloss + self.edge_dis_dloss 
            + self.edge_gloss + self.image_gloss + self.loss_d_ac + self.zl_loss)

    def train(self, train_dl, epochs=100, batch_size=64, device=torch.device("cpu")):
        self.batch_size = batch_size
        optimizer = torch.optim.RMSprop(self.parameters(), lr=1e-4, weight_decay=1e-5)
        start_time = time.time()

        for epoch in range(epochs):
            for idx, data in enumerate(train_dl):
                x, z = data.to(device)
                _ = self.forward(x,z)
                loss = self.compute_loss()
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                discriminator_err = self.joint_dis_dloss + self.image_dis_dloss + self.edge_dis_dloss
                generator_err = self.edge_gloss + self.image_gloss
                print("Epoch: [%2d/%2d] [%4d/%4d], time: %4.4f, joint_dis_dloss: %.8f, joint_dis_gloss: %.8f"
                      % (epoch, epochs, idx, len(train_dl), time.time() - start_time, 2 * discriminator_err, generator_err))