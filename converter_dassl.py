import CoOp.datasets.oxford_pets
import CoOp.datasets.oxford_flowers
import CoOp.datasets.fgvc_aircraft
import CoOp.datasets.dtd
import CoOp.datasets.eurosat
import CoOp.datasets.stanford_cars
import CoOp.datasets.food101
import CoOp.datasets.sun397
import CoOp.datasets.caltech101
import CoOp.datasets.ucf101
import CoOp.datasets.imagenet
import CoOp.datasets.imagenet_sketch
import CoOp.datasets.imagenetv2
import CoOp.datasets.imagenet_a
import CoOp.datasets.imagenet_r

from dassl.config import get_cfg_default
from dassl.utils.tools import read_image
from dassl.data.datasets import build_dataset

from torchvision import transforms
from torch.utils.data import Dataset


CUSTOM_TEMPLATES = {
    "OxfordPets": "a photo of a {}, a type of pet.",
    "OxfordFlowers": "a photo of a {}, a type of flower.",
    "FGVCAircraft": "a photo of a {}, a type of aircraft.",
    "DescribableTextures": "{} texture.",
    "EuroSAT": "a centered satellite photo of {}.",
    "StanfordCars": "a photo of a {}.",
    "Food101": "a photo of {}, a type of food.",
    "SUN397": "a photo of a {}.",
    "Caltech101": "a photo of a {}.",
    "UCF101": "a photo of a person doing {}.",
    "ImageNet": "a photo of a {}.",
    "ImageNetSketch": "a photo of a {}.",
    "ImageNetV2": "a photo of a {}.",
    "ImageNetA": "a photo of a {}.",
    "ImageNetR": "a photo of a {}.",
}


class DasslDataset(Dataset):

    def __init__(self, dassl_dataset, split, transform=None):
        self.samples = []
        self.transform = transform
        self._classnames = dassl_dataset.classnames
        datum_list = {
            "train": dassl_dataset.train_x,
            "val": dassl_dataset.val,
            "test": dassl_dataset.test,
        }[split]
        for data in datum_list:
            self.samples.append((data.impath, data.label))
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, index):
        path, target = self.samples[index]
        img = read_image(path)
        if self.transform is not None:
            img = self.transform(img)
        return img, target
    
    @property
    def classnames(self):
        return self._classnames


def get_raw_dassl_dataset(dataset_name, root, n_shot, subsample="all"):
    cfg = get_cfg_default()
    cfg.DATASET.NAME = dataset_name
    cfg.DATASET.ROOT = root
    cfg.DATASET.NUM_SHOTS = n_shot
    cfg.DATASET.SUBSAMPLE_CLASSES = subsample
    cfg.SEED = 0
    dassl_dataset = build_dataset(cfg)
    return dassl_dataset


def get_dassl_datasets(dataset_name, root, n_shot=0):
    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(224, scale=(0.7, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
    ])
    val_transform = transforms.Compose([
        transforms.Resize(224),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
    ])
    raw_base_dataset = get_raw_dassl_dataset(dataset_name, root, n_shot, "base")
    raw_open_dataset = get_raw_dassl_dataset(dataset_name, root, n_shot, "new")
    template = CUSTOM_TEMPLATES[dataset_name]
    train_dataset = DasslDataset(raw_base_dataset, "train", train_transform)
    val_dataset = DasslDataset(raw_base_dataset, "val", val_transform)
    test_dataset = DasslDataset(raw_base_dataset, "test", val_transform)
    open_dataset = DasslDataset(raw_open_dataset, "test", val_transform)
    base_class_names, open_class_names = train_dataset.classnames, open_dataset.classnames
    return train_dataset, val_dataset, test_dataset, open_dataset, base_class_names, open_class_names, template
