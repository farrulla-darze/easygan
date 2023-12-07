import torch
import torch.nn as nn
import torchvision.utils as vutils


class Generator(nn.Module):
    def __init__(self, Z_DIM, G_HIDDEN, IMAGE_CHANNEL):
        """Generator network.

        Args:
            Z_DIM (int): Size of the latent vector.
            G_HIDDEN (int): Size of the hidden layer.
            IMAGE_CHANNEL (int): Number of channels in the image.
        """
        super(Generator, self).__init__()
        self.main = nn.Sequential(
            # input layer
            nn.ConvTranspose2d(Z_DIM, G_HIDDEN * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(G_HIDDEN * 8),
            nn.ReLU(True),
            # 1st hidden layer
            nn.ConvTranspose2d(G_HIDDEN * 8, G_HIDDEN * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(G_HIDDEN * 4),
            nn.ReLU(True),
            # 2nd hidden layer
            nn.ConvTranspose2d(G_HIDDEN * 4, G_HIDDEN * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(G_HIDDEN * 2),
            nn.ReLU(True),
            # 3rd hidden layer
            nn.ConvTranspose2d(G_HIDDEN * 2, G_HIDDEN, 4, 2, 1, bias=False),
            nn.BatchNorm2d(G_HIDDEN),
            nn.ReLU(True),
            # output layer
            nn.ConvTranspose2d(G_HIDDEN, IMAGE_CHANNEL, 4, 2, 1, bias=False),
            nn.Tanh()
        )

    def forward(self, input):
        return self.main(input)


class Discriminator(nn.Module):
    def __init__(self, IMAGE_CHANNEL, D_HIDDEN):
        """Discriminator network.

        Args:
            IMAGE_CHANNEL (int): Number of channels in the image.
            D_HIDDEN (int): Size of the hidden layer.
        """
        super(Discriminator, self).__init__()
        self.main = nn.Sequential(
            # 1st layer
            nn.Conv2d(IMAGE_CHANNEL, D_HIDDEN, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # 2nd layer
            nn.Conv2d(D_HIDDEN, D_HIDDEN * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(D_HIDDEN * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # 3rd layer
            nn.Conv2d(D_HIDDEN * 2, D_HIDDEN * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(D_HIDDEN * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # 4th layer
            nn.Conv2d(D_HIDDEN * 4, D_HIDDEN * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(D_HIDDEN * 8),
            nn.LeakyReLU(0.2, inplace=True),
            # output layer
            nn.Conv2d(D_HIDDEN * 8, 1, 4, 1, 0, bias=False),
            nn.Sigmoid()
        )

    def forward(self, input):
        return self.main(input).view(-1, 1).squeeze(1)


class GANTrainer:
    def __init__(self, netG, netD, dataloader, device, EPOCH_NUM, REAL_LABEL, FAKE_LABEL, Z_DIM, optimizerG, optimizerD,
                 criterion, viz_noise):
        """GAN Trainer

        Args:
            netG (nn.Module): Generator network.
            netD (nn.Module): Discriminator network.
            dataloader (torch.utils.data.DataLoader): Dataloader.
            device (torch.device): Device to use.
            EPOCH_NUM (int): Number of epochs.
            REAL_LABEL (float): Real label value.
            FAKE_LABEL (float): Fake label value.
            Z_DIM (int): Size of the latent vector.
            optimizerG (torch.optim): Generator optimizer.
            optimizerD (torch.optim): Discriminator optimizer.
            criterion (torch.nn): Loss function.
            viz_noise (torch.Tensor): Fixed noise vector.
        """
        self.netG = netG
        self.netD = netD
        self.dataloader = dataloader
        self.device = device
        self.EPOCH_NUM = EPOCH_NUM
        self.REAL_LABEL = REAL_LABEL
        self.FAKE_LABEL = FAKE_LABEL
        self.Z_DIM = Z_DIM
        self.optimizerG = optimizerG
        self.optimizerD = optimizerD
        self.criterion = criterion
        self.viz_noise = viz_noise

        # Lists to keep track of progress
        self.img_list = []
        self.G_losses = []
        self.D_losses = []
        self.iters = 0

    def update_discriminator_real(self, data):
        """The first step is to train the discriminator network with real data

        Args:
            data (torch.Tensor): Real data.

        Returns:
            errD_real (torch.Tensor): Discriminator loss on real data.
            D_x (float): Discriminator output mean for the batch.
        """
        self.netD.zero_grad()
        real_device = data[0].to(self.device)
        b_size = real_device.size(0)
        label = torch.full((b_size,), self.REAL_LABEL, dtype=torch.float, device=self.device)
        # Forward the real data through the discriminator network
        output = self.netD(real_device).view(-1)
        # Calculate loss
        errD_real = self.criterion(output, label)
        # Backpropagation through the Discriminator net
        errD_real.backward()
        D_x = output.mean().item()  # discriminator output mean for the batch
        return errD_real, D_x

    def update_discriminator_fake(self, data, errD_real):
        """Train the discriminator with fake data

        Args:
            data (torch.Tensor): Real data.
            errD_real (torch.Tensor): Discriminator loss on real data.

        Returns:
            errD (torch.Tensor): Total discriminator loss.
            D_G_z1 (float): Discriminator output mean for the fake batch.
            fake (torch.Tensor): Fake images.
        """
        b_size = self.viz_noise.size(0)
        # Generate a batch of latent space vectors
        noise = torch.randn(b_size, self.Z_DIM, 1, 1, device=self.device)
        # Pass latent variables/noise vector through the generator network in order to generate fake images/data
        fake = self.netG(noise)

        label = torch.full((b_size,), self.FAKE_LABEL, dtype=torch.float, device=self.device)

        # Forward the fake data through the discriminator network
        output = self.netD(fake.detach()).view(-1)
        # Calculate loss
        errD_fake = self.criterion(output, label)
        # Backpropagation through the Discriminator net
        errD_fake.backward()
        D_G_z1 = output.mean().item()  # discriminator output mean for the fake batch
        errD = errD_real + errD_fake  # total discriminator loss
        self.optimizerD.step()  # update discriminator weights
        return errD, D_G_z1, fake

    def update_generator(self, fake):
        """Update the generator network

        Args:
            fake (torch.Tensor): Fake images.

        Returns:
            errG (torch.Tensor): Generator loss.
            D_G_z2 (float): Discriminator output mean for the fake batch.
        """
        self.netG.zero_grad()
        label = torch.full((self.viz_noise.size(0),), self.REAL_LABEL, dtype=torch.float, device=self.device)
        output = self.netD(fake).view(-1)
        # Calculate loss
        errG = self.criterion(output, label)
        # Backpropagation through the Generator net
        errG.backward()
        D_G_z2 = output.mean().item()
        self.optimizerG.step()
        return errG, D_G_z2

    def train(self):
        """Train the GAN model.

        Returns:
            img_list (list): List of fake images.
            G_losses (list): Generator losses.
            D_losses (list): Discriminator losses.
        """
        print("Starting Training Loop...")
        for epoch in range(self.EPOCH_NUM):
            for i, data in enumerate(self.dataloader, 0):
                # Update discriminator with real data
                errD_real, D_x = self.update_discriminator_real(data)

                # Update discriminator with fake data
                errD, D_G_z1, fake = self.update_discriminator_fake(data, errD_real)

                # Update generator
                errG, D_G_z2 = self.update_generator(fake)

                # Output training stats
                if i % 50 == 0:
                    print('[%d/%d][%d/%d]\tLoss_D: %.4f\tLoss_G: %.4f\tD(x): %.4f\tD(G(z)): %.4f / %.4f'
                          % (epoch, self.EPOCH_NUM, i, len(self.dataloader),
                             errD.item(), errG.item(), D_x, D_G_z1, D_G_z2))

                # Save Losses for plotting later
                self.G_losses.append(errG.item())
                self.D_losses.append(errD.item())

                # Check how the generator is doing by saving G's output on fixed_noise
                if (self.iters % 500 == 0) or ((epoch == self.EPOCH_NUM-1) and (i == len(self.dataloader)-1)):
                    with torch.no_grad():
                        fake = self.netG(self.viz_noise).detach().cpu()
                    self.img_list.append(vutils.make_grid(fake, padding=2, normalize=True))

                self.iters += 1
        return self.img_list, self.G_losses, self.D_losses