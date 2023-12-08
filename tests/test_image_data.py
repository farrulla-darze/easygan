import unittest
from torchvision import transforms, utils as vutils
import torchvision.datasets as datasets
from image_data import ImageData
import torch

class TestImageData(unittest.TestCase):
    def setUp(self):
        """Test setup"""
        # Create an instance of the ImageData class
        self.image_data = ImageData()

    def test_get_datasets_list(self):
        """Test get_datasets_list method from torchvision.datasets
            which entails all the possible datasets that can be used
        """
        # Call the get_datasets_list method
        datasets_list = self.image_data.get_datasets_list()

        # Check that the returned value is a list
        self.assertIsInstance(datasets_list,tuple)

    def test_get_dataset(self):
        """Test get_dataset method from torchvision.datasets"""
        # Define parameters for the get_dataset method
        dataset_name = 'MNIST'  # replace with your actual dataset name
        root_dir = '../data/'  # replace with your actual root directory
        transform = transforms.Compose([transforms.ToTensor()])
        download = True

        # Call the get_dataset method
        dataset = self.image_data.get_dataset(dataset_name, root_dir, transform, download)

        # Check that the returned value is a torchvision.datasets object
        self.assertIsInstance(dataset, datasets.MNIST)

    def test_get_dataloader(self):
        """Test get_dataloader method from torchvision.utils.data.DataLoader"""
        # Define parameters for the get_dataloader method
        batch_size = 64  # replace with your actual batch size
        shuffle = True
        num_workers = 2  # replace with your actual number of workers

        # Get a dataset to use for the dataloader
        dataset_name = 'MNIST'  # replace with your actual dataset name
        root_dir = '../data'  # replace with your actual root directory
        transform = transforms.Compose([transforms.ToTensor()])
        download = True
        dataset = self.image_data.get_dataset(dataset_name, root_dir, transform, download)

        # Call the get_dataloader method
        dataloader = self.image_data.get_dataloader(dataset, batch_size, shuffle, num_workers)

        # Check that the returned value is a torch.utils.data.DataLoader object
        self.assertIsInstance(dataloader, vutils.data.DataLoader)

if __name__ == '__main__':
    unittest.main()