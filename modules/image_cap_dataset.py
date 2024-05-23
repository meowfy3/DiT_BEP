import os

import matplotlib.pyplot as plt
from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms import transforms

from modules.training_utils import center_crop_arr


class ImgDataset(Dataset):
    def __init__(self, dataset_path, split='train', transform=None, target_transform=None, res=256):
        self.transform = transform if transform is not None else transforms.Compose([
            transforms.Lambda(lambda pil_image: center_crop_arr(pil_image, res)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], inplace=True)
        ])
        self.dataset_path = dataset_path
        self.target_transform = target_transform
        self.orange = os.listdir(os.path.join(dataset_path, 'Orange'))
        self.blue = os.listdir(os.path.join(dataset_path, 'Blue'))

        # Splitting the dataset
        num_images = len(self.blue)
        train_end = int(0.8 * num_images)
        val_end = int(0.9 * num_images)
        
        if split == 'train':
            self.orange = self.orange[:train_end]
            self.blue = self.blue[:train_end]
        elif split == 'val':
            self.orange = self.orange[train_end:val_end]
            self.blue = self.blue[train_end:val_end]
        elif split == 'test':
            self.orange = self.orange[val_end:]
            self.blue = self.blue[val_end:]
        else:
            raise ValueError("Invalid split name")
    
    def __len__(self):
        return len(self.blue)

    def __getitem__(self, idx):
        path_orange = os.path.join(self.dataset_path, 'Orange', self.orange[idx])
        path_blue = os.path.join(self.dataset_path, 'Blue', self.orange[idx].replace("o", "p"))
        image_target = Image.open(path_blue)
        image_sample = Image.open(path_orange)
        if self.transform:
            image_target = self.transform(image_target)
            image_sample = self.transform(image_sample)
        return image_target, image_sample  # blue, orange


if __name__ == '__main__':
    img1, img2 = next(iter(ImgDataset(r'/workspace/BEP256')))
    plt.imshow(img1.permute(1, 2, 0))
    plt.show()
    plt.imshow(img2.permute(1, 2, 0))
    plt.show()
