import os
import time

import torch
import pandas as pd
from torchvision.datasets.folder import default_loader
from torchvision.datasets.utils import download_url
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, datasets
from PIL import Image


def transform_norm(image_size, mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)):
    """
    Return tensor transform with normalising
    :param image_size: size of resulting images
    :param mean: desired mean of pixels distribution
    :param std: desired standard deviation of pixels distribution
    :return: transforms
    """
    if image_size <= 0:
        raise ValueError('Image size must be a positive number')
    if not isinstance(image_size, int):
        raise TypeError('Image size must be integer')
    return transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])


def transform_unnorm(image_size):
    """
    Return tensor transform without normalising
    :param image_size: size of resulting images
    :return: transforms
    """
    if image_size <= 0:
        raise ValueError('Image size must be a positive number')
    if not isinstance(image_size, int):
        raise TypeError('Image size must be integer')
    return transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
    ])


def load_data(data_dir: str, image_size: int, batch_size: int = 8, is_norm: bool = True):
    """
    Load dataset to Dataloader object
    :param data_dir: directory containing dataset
    :param image_size: size of image
    :param is_norm: bool value indicating if images have to be normalised
    :param batch_size: size of images batch
    :return: dataloader and dataset
    """
    print(f'Started loading dataset from {data_dir}')
    start = time.time()
    transform = transform_norm(image_size) if is_norm else transform_unnorm(image_size)
    dataset = datasets.ImageFolder(data_dir, transform=transform)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    print(f'Successfully loaded dataset in {round(time.time() - start, 2)} min')
    return dataloader, dataset


def load_image(path: str, image_size, device: str = 'cpu'):
    transform = transform_unnorm(image_size)
    content_img = Image.open(path)
    content_img = transform(content_img)
    content_img.unsqueeze_(0)
    return content_img.to(device, torch.float)


def load_cub(parent_dir, image_size, batch_size, is_norm: bool = True):
    """
    Load CUB dataset to Dataloader object
    :param parent_dir: parent directory containing CUB_200_2011 folder
    :param image_size: size of image
    :param batch_size: size of images batch
    :param is_norm: bool value indicating if images have to be normalised
    :return: CUB dataloader and dataset
    """
    print(f'Started loading CUB dataset from {os.path.join(parent_dir, "CUB_200_2011")}')
    start = time.time()
    transform = transform_norm(image_size) if is_norm else transform_unnorm(image_size)
    dataset = Cub2011(parent_dir, train=True, transform=transform, download=False)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    print(f'Successfully loaded dataset in {round((time.time() - start) / 60, 2)} min')
    return dataloader, dataset


class Cub2011(Dataset):
    """
    Cub dataset manager
    """
    base_folder = 'CUB_200_2011/images'
    url = 'http://www.vision.caltech.edu/visipedia-data/CUB-200-2011/CUB_200_2011.tgz'
    filename = 'CUB_200_2011.tgz'
    tgz_md5 = '97eceeb196236b17998738112f37df78'

    def __init__(self, root, train=True, transform=None, loader=default_loader, download=True):
        self.root = os.path.expanduser(root)
        self.transform = transform
        self.loader = default_loader
        self.train = train

        if download:
            self._download()

        if not self._check_integrity():
            raise FileNotFoundError('Dataset not found or corrupted.' +
                                    ' You can use download=True to download it')

    def _load_metadata(self):
        """
        Load necessary files
        """
        images = pd.read_csv(os.path.join(self.root, 'CUB_200_2011', 'images.txt'), sep=' ',
                             names=['img_id', 'filepath'])
        image_class_labels = pd.read_csv(os.path.join(self.root, 'CUB_200_2011', 'image_class_labels.txt'),
                                         sep=' ', names=['img_id', 'target'])
        train_test_split = pd.read_csv(os.path.join(self.root, 'CUB_200_2011', 'train_test_split.txt'),
                                       sep=' ', names=['img_id', 'is_training_img'])
        attributes = pd.read_csv(os.path.join(self.root, 'CUB_200_2011', 'attributes', 'attributes.txt'),
                                 sep=' ', names=['attribute_id', 'attribute'])
        image_attributes = pd.read_csv(
            os.path.join(self.root, 'CUB_200_2011', 'attributes', 'image_attribute_labels.txt'), sep=' ',
            names=['img_id', 'attribute_id', 'is_present', 'certainty_id', 'time'],
            on_bad_lines='skip',
        )
        certainty = pd.read_csv(os.path.join(self.root, 'CUB_200_2011', 'attributes', 'certainties.txt'),
                                sep=' ', names=['certainty_id', 'certainty'])

        attribute_data = image_attributes.merge(certainty, on='certainty_id')
        attribute_data = attribute_data.merge(attributes, on='attribute_id')
        data = images.merge(image_class_labels, on='img_id')
        data = data.merge(train_test_split, on='img_id')
        attribute_data = attribute_data.drop(columns=['attribute_id', 'certainty_id'])
        self.data = data
        self.attributes = attribute_data

        if self.train:
            self.data = self.data[self.data.is_training_img == 1]
        else:
            self.data = self.data[self.data.is_training_img == 0]

    def _check_integrity(self):
        """
        Check if all necessary files are present
        """
        try:
            self._load_metadata()
        except Exception:
            return False

        for index, row in self.data.iterrows():
            filepath = os.path.join(self.root, self.base_folder, row.filepath)
            if not os.path.isfile(filepath):
                print(filepath)
                return False
        return True

    def _download(self):
        """
        Download files needed for CUB
        """
        import tarfile

        if self._check_integrity():
            print('Files already downloaded and verified')
            return

        download_url(self.url, self.root, self.filename, self.tgz_md5)

        with tarfile.open(os.path.join(self.root, self.filename), "r:gz") as tar:
            tar.extractall(path=self.root)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data.iloc[idx]
        path = os.path.join(self.root, self.base_folder, sample.filepath)
        target = sample.target - 1  # Targets start at 1 by default, so shift to 0
        img = self.loader(path)
        img_id = sample.img_id
        attributes = self.attributes[self.attributes.img_id == img_id]
        attributes = attributes[attributes.is_present == 1]
        attributes = attributes[attributes.certainty != '1']
        attributes = attributes['attribute'].tolist()

        if self.transform is not None:
            img = self.transform(img)

        return img, target, ', '.join(attributes)
