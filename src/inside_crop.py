
from torchvision import transforms
import random
import numpy as np
# copy from MoCo https://github.com/facebookresearch/moco
from PIL import ImageFilter
import random


class TwoCropsTransform:
    """Take two random crops of one image as the query and key."""

    def __init__(self, base_transform):
        self.base_transform = base_transform

    def __call__(self, x):
        q = self.base_transform(x)
        k = self.base_transform(x)
        return [q, k]


class GaussianBlur(object):
    """Gaussian blur augmentation in SimCLR https://arxiv.org/abs/2002.05709"""

    def __init__(self, sigma=[.1, 2.]):
        self.sigma = sigma

    def __call__(self, x):
        sigma = random.uniform(self.sigma[0], self.sigma[1])
        x = x.filter(ImageFilter.GaussianBlur(radius=sigma))
        return x
class inside_crop(object):
    def __init__(
            self,
            size_crops,
            nmb_crops,
            min_scale_crops,
            max_scale_crops,normalize):
        assert len(size_crops) == len(nmb_crops)
        assert len(min_scale_crops) == len(nmb_crops)
        assert len(max_scale_crops) == len(nmb_crops)
        self.size_crops = size_crops
        self.nmb_crops = nmb_crops
        trans_weak=[]
        self.sample_choice = 0
        for i in range(len(size_crops)):
            if size_crops[i]==224:
                randomresizedcrop = transforms.RandomResizedCrop(
                    size_crops[i],
                    scale=(min_scale_crops[i], max_scale_crops[i]),
                )
                weak = transforms.Compose([
                    randomresizedcrop,
                    transforms.RandomApply([
                        transforms.ColorJitter(0.8, 0.8, 0.8, 0.2)  # not strengthened
                    ], p=0.8),
                    transforms.RandomGrayscale(p=0.2),
                    transforms.RandomApply([GaussianBlur([.1, 2.])], p=0.5),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    normalize
                ])
                self.sample_choice+=1
            else:
                continue
            trans_weak.extend([weak]*nmb_crops[i])

        self.trans=trans_weak
        print("in total we have %d transforms"%(len(self.trans)))
    def __call__(self, x):
        multi_crops = list(map(lambda trans: trans(x), self.trans))
        extend_crops = []
        #gen random resized crop from the inside of the crops
        for i in range(len(self.size_crops)):
            if self.size_crops[i]==224:
                continue
            else:
                for __ in range(self.nmb_crops[i]):
                    random_choice = random.randint(0,self.sample_choice-1)
                    waiting_crop = multi_crops[random_choice]
                    current_crop_size = self.size_crops[i]
                    channels, height, width = waiting_crop.shape
                    rand_x = random.randint(0, height - current_crop_size)
                    rand_y = random.randint(0, width - current_crop_size)
                    i_bottom = np.clip(rand_x + current_crop_size, 0, height)
                    j_right = np.clip(rand_y + current_crop_size, 0, width)
                    new_crop = waiting_crop[:,rand_x:i_bottom,rand_y:j_right].contiguous()
                    extend_crops.append(new_crop)
        multi_crops.extend(extend_crops)
        return multi_crops