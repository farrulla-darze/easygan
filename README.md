# easyGAN
Simple to use Generative Adversarial Networks (GANs) and PyTorch. It includes a module for handling image data and a module for defining and training GANs.

## Project Structure

- `image_data.py`: This module contains the `ImageData` class for handling image datasets. It includes methods for getting a list of available datasets, getting a specific dataset, and getting a dataloader for a dataset.

- `dcgan.py`: This module contains the `Generator` and `Discriminator` classes for defining the GAN, and the `GANTrainer` class for training the GAN.

- `tests/`: This module contains unit tests for all the modules.


## Requirements

- Python
- PyTorch
- torchvision
- matplotlib
- unittest
- jupyter
- numpy


## Virtual Environment Setup

It's recommended to create a virtual environment to keep the dependencies required by different projects separate and organized. Here's how you can set up a virtual environment for this project:

1. Install the `venv` module, if it's not already installed: `python3 -m pip install --user virtualenv`
2. Navigate to the project directory: `cd <project-directory>`
3. Create a new virtual environment: `python3 -m venv env`
4. Activate the virtual environment:
    - On macOS and Linux: `source env/bin/activate`
    - On Windows: `.\env\Scripts\activate`

Remember to activate the virtual environment every time you work on the project. When you're done, you can deactivate the virtual environment by simply typing `deactivate` in the terminal.

# General Setup
1. Clone the repository.
2. Install the dependencies using pip: `pip install -r requirements.txt`
3. Run the tests to ensure everything is set up correctly: `python -m unittest`

## Usage

1. Import the necessary classes from the modules.
2. Create an instance of the `ImageData` class.
3. Use the `get_datasets_list`, `get_dataset`, and `get_dataloader` methods to handle your image data.
4. Create instances of the `Generator` and `Discriminator` classes.
5. Create an instance of the `GANTrainer` class and use the `train` method to train your GAN.
6. Use the functions presented on `utils` to show images and plots.

## Run the notebook 
Just type:
- jupyter notebook

## Contributing

Contributions are welcome. Please open an issue or submit a pull request.

# New Libraries

If you wish to install new libraries, you just need to install them on your virtual environment and not on your system environment. In other words, to install new libraries in this project, just type:

- pipenv install lib-name

## License

[MIT](https://choosealicense.com/licenses/mit/)
