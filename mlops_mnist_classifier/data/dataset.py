from torch.utils.data import Dataset
from torchvision.transforms import Normalize
from torch import mean, std

# def TransformMnist(x):
#         normalize_x = Normalize(mean=mean(x), std=std(x))
#         return lambda x: normalize_x(x).view(784)


def TransformMnist(x):
    
    normalize_x = Normalize(mean=mean(x), std=std(x))
    x = x.unsqueeze(-1)
    return lambda x: normalize_x(x)


def TransformMnistProcessed(x):
    return lambda x: x


class MyDataset(Dataset):
    def __init__(self, imgs, targets, transform=None):
        self.images = imgs
        self.targets = targets
        self.transform = transform

    def get_saveable_repr(self):
        return self.images, self.targets

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, idx):
        image = self.transform(self.images[idx])
        label = self.targets[idx]
        return image, label
