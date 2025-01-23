import torch
import torchvision.transforms as T
from torch.utils.data import Dataset, ConcatDataset, random_split, Subset
import kagglehub
import os
from PIL import Image
from matplotlib import pyplot as plt
from typing import Literal
from tqdm import tqdm
import numpy as np


def index_to_char(index):
    if index <= 9:
        return chr(index + ord("0"))
    elif 10 <= index <= 35:
        return chr(index - 10 + ord("A"))
    elif 36 <= index <= 61:
        return chr(index - 36 + ord("a"))
    else:
        raise ValueError(f"Invalid index: {index}")


def digits_to_label(digits):
    return "".join([index_to_char(index) for index in digits])


class CaptchaDataset(Dataset):
    def __init__(
        self,
        dataset_handle: Literal[
            "akashguna/large-captcha-dataset", "parsasam/captcha-dataset"
        ],
        label_type: Literal["string", "digits"] = "digits",
        force_download=False,
        transform=None,
        use_cache_when_dropping_corrupted=True,
    ):
        super().__init__()
        self.dataset_handle = dataset_handle
        if dataset_handle == "akashguna/large-captcha-dataset":
            # https://www.kaggle.com/datasets/akashguna/large-captcha-dataset
            self.dataset_path = os.path.join(
                kagglehub.dataset_download(
                    "akashguna/large-captcha-dataset", force_download=force_download
                ),
                "Large_Captcha_Dataset",
            )
        elif dataset_handle == "parsasam/captcha-dataset":
            # https://www.kaggle.com/datasets/parsasam/captcha-dataset/data
            self.dataset_path = kagglehub.dataset_download("parsasam/captcha-dataset")
        elif os.path.isdir(self.dataset_handle):
            self.dataset_path = self.dataset_handle
            use_cache_when_dropping_corrupted = False
        else:
            raise AttributeError(f"Incorrect dataset handle: {dataset_handle}")

        self.paths = os.listdir(self.dataset_path)
        self.transform = transform
        self.label_type = label_type

        self.drop_corrupted_images(use_cached=use_cache_when_dropping_corrupted)

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        img_name = self.paths[index]
        img_path = os.path.join(self.dataset_path, img_name)

        image = Image.open(img_path)
        label = os.path.splitext(img_name)[0]

        if self.transform:
            image = self.transform(image)

        if self.label_type == "string":
            pass
        elif self.label_type == "digits":
            label = self.label_to_digits(label)
        else:
            raise AttributeError(f"Invalid label_type: {self.label_type}")

        return image, label

    def label_to_digits(self, label):
        return torch.tensor([self.char_to_index(c) for c in label], dtype=torch.long)

    def char_to_index(self, char):
        if char.isdigit():  # '0'-'9' -> 0-9
            return int(char)
        elif "A" <= char <= "Z":  # 'A'-'Z' -> 10-35
            return ord(char) - ord("A") + 10
        elif "a" <= char <= "z":  # 'a'-'z' -> 36-61
            return ord(char) - ord("a") + 36
        else:
            raise ValueError(f"Invalid character: {char}")

    def drop_corrupted_images(self, use_cached=True):
        corrupted_images = []
        if use_cached:
            if self.dataset_handle == "parsasam/captcha-dataset":
                corrupted_images = []
            elif self.dataset_handle == "akashguna/large-captcha-dataset":
                corrupted_images = ["4q2wA.png"]
        else:
            print("looking for corrupted images")
            for img_name in tqdm(self.paths):
                img_path = os.path.join(self.dataset_path, img_name)
                try:
                    Image.open(img_path).verify()
                except:
                    corrupted_images.append(img_name)

        for corrupted_image in corrupted_images:
            self.paths.remove(corrupted_image)


def get_combined_dataset(transform=None):
    dset1 = CaptchaDataset(
        "akashguna/large-captcha-dataset",
        transform=transform,
    )
    dset2 = CaptchaDataset(
        "parsasam/captcha-dataset",
        transform=transform,
    )

    return ConcatDataset((dset1, dset2))


def split_dataset(dataset, splits=[0.8, 0.1, 0.1], index_cache_path=None):
    if (not index_cache_path is None) and os.path.isfile(index_cache_path):
        indices = torch.load(index_cache_path)
    else:
        indices = random_split(torch.arange(len(dataset)), splits)
        if index_cache_path is not None:
            torch.save(indices, index_cache_path)

    sets = []
    for split_indices in indices:
        sets.append(Subset(dataset, split_indices))

    return sets, indices


if __name__ == "__main__":
    # dataset = CaptchaDataset(
    #     "parsasam/captcha-dataset",
    #     transform=T.Compose([T.Resize((512, 512)), T.ToTensor()]),
    # )
    # dataset = get_combined_dataset(T.Compose([T.Resize((512, 512)), T.ToTensor()]))
    # x, y = dataset[130000]
    # print(len(dataset))
    # print(y.shape)
    # plt.imshow(torch.permute(x, (1, 2, 0)))
    # plt.savefig("image.png")

    dataset = get_combined_dataset(T.Compose([T.Resize((512, 512)), T.ToTensor()]))
    splits, indices = split_dataset(
        dataset, [0.8, 0.1, 0.1], "combined_dset_splits_index_cache.pt"
    )

    print(len(splits))
    print([len(split) for split in splits])
    print(len(indices))
    print([len(index) for index in indices])
    print(indices[0][0])
    print(indices[1][0])
    print(indices[2][0])
