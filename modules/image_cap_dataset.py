import os

import matplotlib.pyplot as plt
from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms import transforms

from modules.training_utils import center_crop_arr


class ImgDataset(Dataset):
    def __init__(self, dataset_path, transform=None, target_transform=None, res=256):
        self.transform = transform if transform is not None else transforms.Compose([
            transforms.Lambda(lambda pil_image: center_crop_arr(pil_image, res)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], inplace=True)
        ])
        self.dataset_path = dataset_path
        self.target_transform = target_transform
        self.orange = os.listdir(os.path.join(dataset_path, 'Orange'))

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
    img1, img2 = next(iter(ImgDataset('/content/drive/My Drive/BEP_Datasets/256/')))
    plt.imshow(img1.permute(1, 2, 0))
    plt.show()
    plt.imshow(img2.permute(1, 2, 0))
    plt.show()
