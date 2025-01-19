import torch
import torchvision.transforms as T
from torch.utils.data import Dataset, ConcatDataset
import kagglehub
import os
from PIL import Image
from matplotlib import pyplot as plt
from typing import Literal


class CaptchaDataset(Dataset):
    def __init__(
        self,
        dataset_handle: Literal[
            "akashguna/large-captcha-dataset", "parsasam/captcha-dataset"
        ],
        label_type: Literal["string", "digits"] = "digits",
        force_download=False,
        transform=None,
    ):
        super().__init__()
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
        else:
            raise AttributeError(f"Incorrect dataset handle: {dataset_handle}")

        self.paths = os.listdir(self.dataset_path)
        self.transform = transform
        self.label_type = label_type

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


if __name__ == "__main__":
    # dataset = CaptchaDataset(
    #     "parsasam/captcha-dataset",
    #     transform=T.Compose([T.Resize((512, 512)), T.ToTensor()]),
    # )
    dataset = get_combined_dataset(T.Compose([T.Resize((512, 512)), T.ToTensor()]))
    x, y = dataset[130000]
    print(len(dataset))
    print(y.shape)
    plt.imshow(torch.permute(x, (1, 2, 0)))
    plt.savefig("image.png")
