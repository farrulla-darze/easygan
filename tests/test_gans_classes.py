import unittest
import torch
from dcgan import Generator

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

if __name__ == '__main__':
    unittest.main()