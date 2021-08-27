import re
from torch.utils.data import Dataset, DataLoader
from numpy import random
import numpy as np
import cv2 as cv
from pymongo import MongoClient

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
        client = MongoClient('mongodb://127.0.0.1', 27017)
        self.data = list(client.wallhaven_v2.wallpaper.find())
        # self.aug = PhotometricDistort()
        random.shuffle(self.data)

    def __getitem__(self, index):
        record = self.data[index]
        img = cv.imread(record['train_path'])
        # if random.randint(2):
        #     img = cv.resize(img, (img.shape[1], int(img.shape[0]*random.uniform(0.9, 1.1))))
        # if random.randint(2):
        #     img = cv.resize(img, (int(img.shape[1]*random.uniform(0.9, 1.1)), img.shape[0]))
        if random.randint(2):
            cv.flip(img, 1)
        img = img.astype(np.float32)
        # h, w, _ = img.shape
        # if h>w:
        #     d = (h-w)//2
        #     img = cv.copyMakeBorder(img,0,0,d,d,cv.BORDER_CONSTANT,value=(0,0,0))
        # elif h<w:
        #     d = (w-h)//2
        #     img = cv.copyMakeBorder(img,d,d,0,0,cv.BORDER_CONSTANT,value=(0,0,0))
        # img = cv.resize(img, (512,512))

        embeddings = record['embeddings']
        embedding_ids = record['embedding_index']
        labels = np.zeros((56, 1, 768), dtype=np.float32)
        masks = np.zeros((56), dtype=np.float32)
        for n, i in enumerate(embedding_ids):
            labels[i] = embeddings[n]
            masks[i] = 1

        return img, labels, masks

    def __len__(self):
        return len(self.data)


def get_dataloader(batch_size, n):
    return DataLoader(Wallpaper(), batch_size, True, num_workers=n, pin_memory=True)

if __name__ == '__main__':
    loader = get_dataloader(2, 1)
    for img, labels, masks in loader:
        print(img.shape, labels.shape, masks.shape)
        print((labels*masks.unsqueeze(-1).unsqueeze(-1)).shape)