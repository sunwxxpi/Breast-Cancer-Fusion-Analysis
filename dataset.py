import os
from PIL import Image
from torch.utils.data import Dataset


class CustomDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.classes = sorted(os.listdir(os.path.join(root_dir, 'elastogram_gray')))
        self.image_paths = []
        self.labels = []

        for label, class_name in enumerate(self.classes):
            elastogram_gray_dir = os.path.join(root_dir, 'elastogram_gray', class_name)
            elastogram_dir = os.path.join(root_dir, 'elastogram', class_name)
            elastogram_images = os.listdir(elastogram_dir)
            for image_name in elastogram_images:
                self.image_paths.append((os.path.join(elastogram_gray_dir, image_name),
                                        os.path.join(elastogram_dir, image_name)))
                self.labels.append(label)

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        elastogram_gray_path, elastogram_path = self.image_paths[idx]
        elastogram_gray_image = Image.open(elastogram_gray_path).convert('RGB')
        elastogram_image = Image.open(elastogram_path).convert('RGB')
        label = self.labels[idx]

        if self.transform:
            elastogram_gray_image = self.transform(elastogram_gray_image)
            elastogram_image = self.transform(elastogram_image)

        return elastogram_gray_image, elastogram_image, label


class BUSIDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.classes = sorted(os.listdir(os.path.join(root_dir, 'busi_elastogram_gray')))
        self.image_paths = []
        self.labels = []

        for label, class_name in enumerate(self.classes):
            elastogram_gray_dir = os.path.join(root_dir, 'busi_elastogram_gray', class_name)
            elastogram_dir = os.path.join(root_dir, 'busi_elastogram', class_name)
            elastogram_images = os.listdir(elastogram_dir)
            for image_name in elastogram_images:
                self.image_paths.append((os.path.join(elastogram_gray_dir, image_name), 
                                         os.path.join(elastogram_dir, image_name)))
                self.labels.append(label)

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        elastogram_gray_path, elastogram_path = self.image_paths[idx]
        elastogram_gray_image = Image.open(elastogram_gray_path).convert('RGB')
        elastogram_image = Image.open(elastogram_path).convert('RGB')
        label = self.labels[idx]

        if self.transform:
            elastogram_gray_image = self.transform(elastogram_gray_image)
            elastogram_image = self.transform(elastogram_image)

        return elastogram_gray_image, elastogram_image, label