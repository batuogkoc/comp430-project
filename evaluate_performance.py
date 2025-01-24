import torch
from torch.utils.data import DataLoader
import torchvision.transforms as T
from train import ResNetWrapper, ResNetWrapperDifferentiable
from captcha_datasets.datasets import CaptchaDataset, index_to_char, digits_to_label
from matplotlib import pyplot as plt
from captcha_datasets.datasets import *
from torch import nn
from tqdm import tqdm


# def load_checkpoint(path):
#     checkpoint = torch.load(path, weights_only=False)

#     model = ResNetWrapperDifferentiable(5, 62, "resnet-101")
#     model.load_state_dict(checkpoint["model_state_dict"])

#     loss_fn = torch.nn.CrossEntropyLoss()
#     return model, loss_fn

def load_checkpoint(path, device):
    checkpoint = torch.load(path, map_location=device)

    model = ResNetWrapperDifferentiable(5, 62, "resnet-101")
    model.load_state_dict(checkpoint["model_state_dict"])

    loss_fn = torch.nn.CrossEntropyLoss()
    return model, loss_fn


def evaluate_performance(dataset_loader: DataLoader, model: nn.Module, device):
    model.to(device)
    model.eval()
    total_accuracy = 0
    per_digit_accuracy = 0
    n = 0
    for x, y in tqdm(dataset_loader):
        x = x.to(device)
        y = y.to(device)

        y_pred = model(x)
        correct_predictions = (torch.argmax(y_pred, 1) == y).to(torch.float32)
        total_accuracy += torch.sum(torch.all(correct_predictions, dim=-1))
        per_digit_accuracy += torch.sum(torch.mean(correct_predictions, dim=-1))
        n += correct_predictions.shape[0]

    return total_accuracy / n, per_digit_accuracy / n


if __name__ == "__main__":
    # path to the cleaned_checkpoint_1.pt file
    #CHECKPOINT_PATH = "/Users/idilgorgulu/Desktop/cleaned_checkpoint_1.pt"
    CHECKPOINT_PATH = "cleaned_checkpoint_1.pt"
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    #DEVICE = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
    print("Using device:", DEVICE)
    model, loss_fn = load_checkpoint(CHECKPOINT_PATH, DEVICE)
    transforms = T.Compose(
        (
            T.Resize((224, 224)),
            T.ToTensor(),
        )
    )

    # load one of the kaggle datasets
    # dataset = CaptchaDataset(
    #     "akashguna/large-captcha-dataset",
    #     transform=transforms,
    # )

    # load the original train, validation and test sets used in the training
    # (train_set, val_set, test_set), _ = split_dataset(
    #     get_combined_dataset(transforms),
    #     [0.8, 0.1, 0.1],
    #     "combined_splits_indices.cache",
    # )
    # print(len(train_set), len(val_set), len(test_set))

    # load a custom dataset by specifiying the folder (must contain only images, and the name of the image must be the label of the captcha)
    dataset = CaptchaDataset(
        #"/home/bgunduz21/.cache/kagglehub/datasets/parsasam/captcha-dataset/versions/1",
        "perturbed_images/patch",
        transform=transforms,
    )

    loader = DataLoader(dataset, batch_size=16, shuffle=False)
    x, y = next(iter(loader))

    total_accuracy, per_digit_accuracy = evaluate_performance(loader, model, DEVICE)
    print("total accuracy: ", total_accuracy)
    print("per digit average accuracy: ", per_digit_accuracy)
