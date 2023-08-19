# importing libraries
import yaml
import numpy as np
import pandas as pd
import os
import random
import cv2
import torch
from torchvision import transforms, datasets
from torch.utils.data import Dataset

config_path = "config/preprocessing_config.yaml"
conf = yaml.safe_load(open(config_path, "r"))

columns = conf["train_dataset"]["columns"]
image_size = conf["image_preprocessing"]["image_size"]
device = conf["metadata"]["device"]


# def prepare_image(path, sigmaX = 10, do_random_crop = False):
def prepare_image(
    path, sigmaX=10, do_random_crop=False, image_size=224, transform_image=False
):
    # import image
    image = cv2.imread(path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # perform smart crops
    image = crop_black(image, tol=7)
    if do_random_crop == True:
        image = random_crop(image, size=(0.9, 1))

    # resize and color
    image = cv2.resize(image, (int(image_size), int(image_size)))
    image = cv2.addWeighted(image, 4, cv2.GaussianBlur(image, (0, 0), sigmaX), -4, 128)

    # circular crop
    image = circle_crop(image, sigmaX=sigmaX)

    # transform
    if transform_image:
        image = transform(image)
    # convert to tensor
    image = torch.tensor(image).float()
    
    if transform_image:
        image = image.unsqueeze(0)
    
    if not transform_image:
        image = image.permute(2, 1, 0)

    return image


def crop_black(img, tol=7):
    if img.ndim == 2:
        mask = img > tol
        return img[np.ix_(mask.any(1), mask.any(0))]

    elif img.ndim == 3:
        gray_img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        mask = gray_img > tol
        check_shape = img[:, :, 0][np.ix_(mask.any(1), mask.any(0))].shape[0]

        if check_shape == 0:
            return img
        else:
            img1 = img[:, :, 0][np.ix_(mask.any(1), mask.any(0))]
            img2 = img[:, :, 1][np.ix_(mask.any(1), mask.any(0))]
            img3 = img[:, :, 2][np.ix_(mask.any(1), mask.any(0))]
            img = np.stack([img1, img2, img3], axis=-1)
            return img


def circle_crop(img, sigmaX=10):
    height, width, depth = img.shape

    largest_side = np.max((height, width))
    img = cv2.resize(img, (largest_side, largest_side))

    height, width, depth = img.shape

    x = int(width / 2)
    y = int(height / 2)
    r = np.amin((x, y))

    circle_img = np.zeros((height, width), np.uint8)
    cv2.circle(circle_img, (x, y), int(r), 1, thickness=-1)

    img = cv2.bitwise_and(img, img, mask=circle_img)
    return img


def random_crop(img, size=(0.9, 1)):
    height, width, depth = img.shape

    cut = 1 - random.uniform(size[0], size[1])

    i = random.randint(0, int(cut * height))
    j = random.randint(0, int(cut * width))
    h = i + int((1 - cut) * height)
    w = j + int((1 - cut) * width)

    img = img[i:h, j:w, :]

    return img


def transform(image):
    # weighted grayscale
    r_weight = 0.5
    g_weight = 0.9
    b_weight = 0.7
    image = (
        image[:, :, 0] * b_weight
        + image[:, :, 1] * g_weight
        + image[:, :, 2] * r_weight
    )
    image = np.stack([image] * 3, axis=-1)
    image = cv2.convertScaleAbs(image)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # threasholded resize
    ret, image = cv2.threshold(image, 210, 255, cv2.THRESH_BINARY)
    image = cv2.resize(image, (28, 28))
    return image


class EyeData(Dataset):
    # initialize
    def __init__(
        self,
        data,
        directory,
        columns,
        transform=None,
        do_random_crop=True,
        itype=".jpg",
        image_size=224,
        device="cpu",
        reduce=False,
    ):
        self.data = data
        self.directory = directory
        self.columns = columns
        self.transform = transform
        self.do_random_crop = do_random_crop
        self.itype = itype
        self.image_size = image_size
        self.device = device
        self.reduce = reduce

    # length
    def __len__(self):
        return len(self.data)

    # get items
    def __getitem__(self, idx):
        img_name = os.path.join(
            self.directory, self.data.loc[idx, self.columns[0]] + self.itype
        )
        image = prepare_image(
            img_name,
            do_random_crop=self.do_random_crop,
            image_size=self.image_size,
            transform_image= self.reduce,
        )
        if not self.reduce:
            image = self.transform(image)
        label = torch.tensor(self.data.loc[idx, self.columns[1]])
        # if(device == 'cuda:0'):
        #     image = image.to(device)
        #     label = label.to(device)
        return {"image": image, "label": label}
