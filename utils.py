import matplotlib.pyplot as plt
import numpy as np
import torchvision.utils as vutils
import torch.nn as nn
import torch.optim as optim


def weights_init(m):
    """Initialize weights of the network.

    Args:
        m (nn.Module): Network to be initialized.
    """

    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)


def get_real_images(dataloader, device):
    """Get real images from dataloader.

    Args:
        dataloader (torch.utils.data.DataLoader): Dataloader.
        device (torch.device): Device to use.

    Returns:
        real_batch (torch.Tensor): Real images.
    """
    real_batch = next(iter(dataloader))
    plt.figure(figsize=(8, 8))
    plt.axis("off")
    plt.title("Training Images")
    plt.imshow(
        np.transpose(vutils.make_grid(real_batch[0].to(device)[:64], padding=2, normalize=True).cpu(), (1, 2, 0)))

def get_fake_images(image_list):
    """Get fake images from generator.

    Args:
        image_list (list): List of fake images.

    Returns:
        fake (torch.Tensor): Fake images.
    """
    # Select a subset of images
    subset_image_list = image_list[:64]

    # Create a new figure
    plt.figure(figsize=(8, 8))

    # Plot the fake images from the last epoch of the subset in an 8x8 grid
    for i in range(64):
        plt.subplot(8, 8, i + 1)
        plt.axis("off")
        plt.imshow(np.transpose(subset_image_list[i], (1, 2, 0)))

    plt.suptitle("Fake Images")
    plt.show()

def make_plot(G_losses, D_losses):
    """Plot generator and discriminator losses.

    Args:
        G_losses (list): Generator losses.
        D_losses (list): Discriminator losses.

    """
    plt.figure(figsize=(10, 5))
    plt.title("Generator and Discriminator Loss During Training")
    plt.plot(G_losses, label="G")
    plt.plot(D_losses, label="D")
    plt.xlabel("iterations")
    plt.ylabel("Loss")
    plt.legend()
    plt.show()


def set_criterion(criterion_name):
    """Set criterion for training.

    Args:
        criterion_name (str): Name of the criterion.

    Returns:
        criterion (torch.nn): Criterion for training.
    """
    if criterion_name == 'BCE':
        criterion = nn.BCELoss()
    elif criterion_name == 'MSE':
        criterion = nn.MSELoss()
    elif criterion_name == 'L1':
        criterion = nn.L1Loss()
    elif criterion_name == 'SmoothL1':
        criterion = nn.SmoothL1Loss()
    elif criterion_name == 'cross_entropy':
        criterion = nn.CrossEntropyLoss()
    else:
        raise NotImplementedError
    return criterion


def set_optimizer(optimizer_name, net, lr, betas):
    """Set optimizer for training.

    Args:
        optimizer_name (str): Name of the optimizer.
        net (nn.Module): Network to be trained.
        lr (float): Learning rate.
        betas (tuple): Betas for the optimizer.

    Returns:
        optimizer (torch.optim): Optimizer for training.
    """
    if optimizer_name == 'Adam':
        optimizer = optim.Adam(net.parameters(), lr=lr, betas=betas)
    elif optimizer_name == 'SGD':
        optimizer = optim.SGD(net.parameters(), lr=lr, momentum=betas[0])
    elif optimizer_name == 'RMSprop':
        optimizer = optim.RMSprop(net.parameters(), lr=lr, momentum=betas[0])
    else:
        raise NotImplementedError
    return optimizer