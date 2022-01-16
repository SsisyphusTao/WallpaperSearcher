from re import escape
import torch
from torch.utils.data import Dataset, DataLoader
from numpy import random
import numpy as np
import cv2 as cv
from pymongo import MongoClient
import os

class SwapChannels(object):
    """Transforms a tensorized image by swapping the channels in the order
     specified in the swap tuple.
    Args:
        swaps (int triple): final order of channels
            eg: (2, 1, 0)
    """

    def __init__(self, swaps):
        self.swaps = swaps

    def __call__(self, image):
        """
        Args:
            image (Tensor): image tensor to be transformed
        Return:
            a tensor with channels swapped according to swap
        """
        # if torch.is_tensor(image):
        #     image = image.data.cpu().numpy()
        # else:
        #     image = np.array(image)
        image = image[:, :, self.swaps]
        return image

class Compose(object):
    """Composes several augmentations together.
    Args:
        transforms (List[Transform]): list of transforms to compose.
    Example:
        >>> augmentations.Compose([
        >>>     transforms.CenterCrop(10),
        >>>     transforms.ToTensor(),
        >>> ])
    """

    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, img):
        for t in self.transforms:
            img = t(img)
        return img

class RandomContrast(object):
    def __init__(self, lower=0.5, upper=1.5):
        self.lower = lower
        self.upper = upper
        assert self.upper >= self.lower, "contrast upper must be >= lower."
        assert self.lower >= 0, "contrast lower must be non-negative."

    # expects float image
    def __call__(self, image):
        if random.randint(2):
            alpha = random.uniform(self.lower, self.upper)
            image *= alpha
        return image

class ConvertColor(object):
    def __init__(self, current='BGR', transform='HSV'):
        self.transform = transform
        self.current = current

    def __call__(self, image):
        if self.current == 'BGR' and self.transform == 'HSV':
            image = cv.cvtColor(image, cv.COLOR_BGR2HSV)
        elif self.current == 'HSV' and self.transform == 'BGR':
            image = cv.cvtColor(image, cv.COLOR_HSV2BGR)
        else:
            raise NotImplementedError
        return image

class RandomSaturation(object):
    def __init__(self, lower=0.5, upper=1.5):
        self.lower = lower
        self.upper = upper
        assert self.upper >= self.lower, "contrast upper must be >= lower."
        assert self.lower >= 0, "contrast lower must be non-negative."

    def __call__(self, image):
        if random.randint(2):
            image[:, :, 1] *= random.uniform(self.lower, self.upper)

        return image

class RandomHue(object):
    def __init__(self, delta=18.0):
        assert delta >= 0.0 and delta <= 360.0
        self.delta = delta

    def __call__(self, image):
        if random.randint(2):
            image[:, :, 0] += random.uniform(-self.delta, self.delta)
            image[:, :, 0][image[:, :, 0] > 360.0] -= 360.0
            image[:, :, 0][image[:, :, 0] < 0.0] += 360.0
        return image

class RandomBrightness(object):
    def __init__(self, delta=32):
        assert delta >= 0.0
        assert delta <= 255.0
        self.delta = delta

    def __call__(self, image):
        if random.randint(2):
            delta = random.uniform(-self.delta, self.delta)
            image += delta
        return image

class RandomLightingNoise(object):
    def __init__(self):
        self.perms = ((0, 1, 2), (0, 2, 1),
                      (1, 0, 2), (1, 2, 0),
                      (2, 0, 1), (2, 1, 0))

    def __call__(self, image):
        if random.randint(2):
            swap = self.perms[random.randint(len(self.perms))]
            shuffle = SwapChannels(swap)  # shuffle channels
            image = shuffle(image)
        return image

class PhotometricDistort(object):
    def __init__(self):
        self.pd = [
            RandomContrast(),
            ConvertColor(transform='HSV'),
            RandomSaturation(),
            RandomHue(),
            ConvertColor(current='HSV', transform='BGR'),
            RandomContrast()
        ]
        self.rand_brightness = RandomBrightness()
        self.rand_light_noise = RandomLightingNoise()

    def __call__(self, image):
        im = image.copy()
        im = self.rand_brightness(im)
        if random.randint(2):
            distort = Compose(self.pd[:-1])
        else:
            distort = Compose(self.pd[1:])
        im = distort(im)
        return im

class Wallpaper(Dataset):
    def __init__(self) -> None:
        super().__init__()
        client = MongoClient('mongodb://172.17.0.1', 27017)
        self.data = list(client.wallpaper.trainset.find())
        # self.aug = PhotometricDistort()
        # random.shuffle(self.data)

    def __getitem__(self, index):
        record = self.data[index]
        img = cv.imread(record['path'])
        if random.randint(2):
            img = cv.resize(img, (img.shape[1], int(img.shape[0]*random.uniform(0.9, 1.1))))
        if random.randint(2):
            img = cv.resize(img, (int(img.shape[1]*random.uniform(0.9, 1.1)), img.shape[0]))
        if random.randint(2):
            cv.flip(img, 1)
        img = img.astype(np.float32)
        h, w, _ = img.shape
        if h>w:
            d = int(h-w)
            x = random.randint(d)
            y = d-x
            img = cv.copyMakeBorder(img,0,0,x,y,cv.BORDER_CONSTANT,value=(0,0,0))
        elif h<w:
            d = int(w-h)
            x = random.randint(d)
            y = d-x
            img = cv.copyMakeBorder(img,x,y,0,0,cv.BORDER_CONSTANT,value=(0,0,0))
        img = cv.resize(img, (512,512))

        labels = np.array(record['label'], dtype=np.float32)
        return img, labels

    def __len__(self):
        return len(self.data)


def get_dataloader(batch_size, local_rank):
    dataset = Wallpaper()
    batch_size = min(batch_size, len(dataset))
    nw = min([os.cpu_count() // torch.distributed.get_world_size(), batch_size if batch_size > 1 else 0, 8])  # number of workers
    train_sampler = torch.utils.data.distributed.DistributedSampler(dataset) if local_rank != -1 else None
    return DataLoader(dataset, batch_size, sampler=train_sampler, num_workers=2, pin_memory=True)

if __name__ == '__main__':
    loader = get_dataloader(2, 1)
    for img, labels, masks in loader:
        print(img.shape, labels.shape, masks.shape)
        print((labels*masks.unsqueeze(-1).unsqueeze(-1)).shape)