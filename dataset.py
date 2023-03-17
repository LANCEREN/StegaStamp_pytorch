import os
import numpy as np
from glob import glob
from PIL import Image, ImageOps
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import torch


class StegaData(Dataset):
    def __init__(self, data_path, secret_size=100, size=(400, 400)):
        self.data_path = data_path
        self.secret_size = secret_size
        self.size = size
        self.files_list = glob(os.path.join(self.data_path, '*.jpg'))
        self.to_tensor = transforms.ToTensor()
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                std=[0.229, 0.224, 0.225]),
            ])

    def __getitem__(self, idx):
        if len(self.files_list) == 0:
            raise "dataset is empty!"
        img_cover_path = self.files_list[idx]

        img_cover = Image.open(img_cover_path).convert('RGB')
        img_cover = ImageOps.fit(img_cover, self.size)
        img_cover = self.to_tensor(img_cover)
        # img_cover = self.transform(img_cover)
        # img_cover = np.array(img_cover, dtype=np.float32) / 255.

        secret = np.random.binomial(1, 0.5, self.secret_size)
        secret = torch.from_numpy(secret).float()

        return img_cover, secret

    def __len__(self):
        return len(self.files_list)


if __name__ == '__main__':
    # dataset = StegaData(data_path='F:\\VOCdevkit\\VOC2012\\JPEGImages')
    # print(len(dataset))
    # img_cover, secret = dataset[10]
    # print(type(img_cover), type(secret))
    # print(img_cover.shape, secret.shape)

    dataset = StegaData(data_path=r'/mnt/ext/renge/medium-imagenet-data/train/n02493793', secret_size=100, size=(400, 400))
    dataloader = DataLoader(dataset, batch_size=4, shuffle=True, pin_memory=True)
    image_input, secret_input = next(iter(dataloader))
    print(type(image_input), type(secret_input))
    print(image_input.shape, secret_input.shape)
    print(image_input.max())
