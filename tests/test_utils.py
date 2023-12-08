import unittest
from utils import set_criterion, make_plot, get_fake_images, get_real_images
import torchvision.datasets as dset
from torchvision.transforms import transforms
import torch.nn as nn
import torch
from unittest.mock import patch


class TestSetCriterion(unittest.TestCase):
    def test_set_criterion_BCE(self):
        criterion = set_criterion('BCE')
        self.assertIsInstance(criterion, nn.BCELoss)

    def test_set_criterion_MSE(self):
        criterion = set_criterion('MSE')
        self.assertIsInstance(criterion, nn.MSELoss)

    def test_set_criterion_L1(self):
        criterion = set_criterion('L1')
        self.assertIsInstance(criterion, nn.L1Loss)

    def test_set_criterion_SmoothL1(self):
        criterion = set_criterion('SmoothL1')
        self.assertIsInstance(criterion, nn.SmoothL1Loss)

    def test_set_criterion_cross_entropy(self):
        criterion = set_criterion('cross_entropy')
        self.assertIsInstance(criterion, nn.CrossEntropyLoss)

    def test_set_criterion_not_implemented(self):
        with self.assertRaises(NotImplementedError):
            set_criterion('not_implemented')


class TestMakePlot(unittest.TestCase):
    @patch('matplotlib.pyplot.show')
    def test_make_plot(self, mock_show):
        G_losses = [0.1, 0.2, 0.3, 0.4, 0.5]
        D_losses = [0.5, 0.4, 0.3, 0.2, 0.1]
        try:
            make_plot(G_losses, D_losses)
        except Exception as e:
            self.fail(f'make_plot raised {type(e)} unexpectedly!')


class TestGetFakeImages(unittest.TestCase):
    @patch('matplotlib.pyplot.show')
    def test_get_fake_images(self, mock_show):
        image_list = [0.1, 0.2, 0.3, 0.4, 0.5]
        try:
            get_fake_images(image_list)
        except Exception as e:
            self.fail(f'get_fake_images raised {type(e)} unexpectedly!')


class TestGetRealImages(unittest.TestCase):
    @patch('matplotlib.pyplot.show')
    def test_get_real_images(self, mock_show):
        X_DIM = 64  # replace with your actual X_DIM
        BATCH_SIZE = 128  # replace with your actual BATCH_SIZE
        dataset = dset.MNIST(root="./datatest", download=True,
                             transform=transforms.Compose([
                                 transforms.Resize(X_DIM),
                                 transforms.ToTensor(),
                                 transforms.Normalize((0.5,), (0.5,))
                             ]))
        print(dataset)

        # Dataloader
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=BATCH_SIZE,
                                                 shuffle=True, num_workers=2)
        device = 'cpu'
        try:
            get_real_images(dataloader, device)
        except Exception as e:
            self.fail(f'get_real_images raised {type(e)} unexpectedly!')

if __name__ == '__main__':
    unittest.main()
