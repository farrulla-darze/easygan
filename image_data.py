import torchvision.datasets as datasets
from torchvision import transforms, utils as vutils

class ImageData:
    def __init__(self):
        print('ImageData class created')

    def get_datasets_list(self):
        """Get datasets list.

        Returns:
            datasets_list (list): List of datasets.
        """
        datasets_list = datasets.__all__
        return datasets_list

    def get_dataset(self, dataset_name, root_dir, transform, download):
        """Get dataset.

        Args:
            dataset_name (str): Dataset name.
            root_dir (str): Root directory.
            transform (torchvision.transforms): Transformations to be applied to the data.
            download (bool): Download the dataset if True.

        Returns:
            dataset (torchvision.datasets): Dataset.
        """
        dataset = getattr(datasets, dataset_name)(root=root_dir, transform=transform, download=download)
        return dataset

    def get_dataloader(self, dataset, batch_size, shuffle, num_workers):
        """Get dataloader.

        Args:
            dataset (torchvision.datasets): Dataset.
            batch_size (int): Batch size.
            shuffle (bool): Shuffle the data.
            num_workers (int): Number of subprocesses to use for data loading.

        Returns:
            dataloader (torch.utils.data.DataLoader): Dataloader.
        """
        dataloader = vutils.data.DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
        return dataloader




if __name__ == "__main__":
    image_data = ImageData()
    datasets_list = image_data.get_datasets_list()
    print(datasets_list)