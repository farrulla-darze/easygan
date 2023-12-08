import unittest
import torch
from dcgan import Generator, Discriminator

class TestGenerator(unittest.TestCase):
    def test_forward(self):
        Z_DIM = 100  # replace with your actual Z_DIM
        G_HIDDEN = 64  # replace with your actual G_HIDDEN
        IMAGE_CHANNEL = 1  # replace with your actual IMAGE_CHANNEL

        # Create an instance of the Generator class
        generator = Generator(Z_DIM, G_HIDDEN, IMAGE_CHANNEL)

        # Create a test input
        test_input = torch.randn(1, Z_DIM, 1, 1)

        # Call the forward method
        output = generator.forward(test_input)

        # Check the output shape
        self.assertEqual(output.shape, (1, IMAGE_CHANNEL, 64, 64))


class TestDiscriminator(unittest.TestCase):
    def test_forward(self):
        D_HIDDEN = 64  # replace with your actual D_HIDDEN
        IMAGE_CHANNEL = 1  # replace with your actual IMAGE_CHANNEL

        # Create an instance of the Discriminator class
        discriminator = Discriminator(D_HIDDEN, IMAGE_CHANNEL)

        # Create a test input
        test_input = torch.randn(1, IMAGE_CHANNEL, 64, 64)
        print(test_input.shape)
        print(test_input.ndim)

        # Call the forward method
        output = discriminator.forward(test_input)
        print(output.shape)

        # Check the output shape
        self.assertEqual(output.shape, (1, 1))

if __name__ == '__main__':
    unittest.main()
