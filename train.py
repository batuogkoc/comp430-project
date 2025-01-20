import torch
import torchmetrics.classification
from torchvision import transforms as T
from torch.utils.data import DataLoader, random_split, Subset, ConcatDataset, Dataset
from datetime import datetime
import wandb
import torchmetrics
import os
from matplotlib import pyplot as plt
import numpy as np
from torch import optim
from training.train_loop import train
from captcha_datasets.datasets import *
from transformers import AutoImageProcessor, ResNetModel
from torch import nn


class ResNetWrapper(nn.Module):
    def __init__(self, num_digits, num_chars, type="resnet-50"):
        super().__init__()
        self.image_processor = AutoImageProcessor.from_pretrained(
            f"microsoft/{type}", use_fast=True
        )
        self.model = ResNetModel.from_pretrained(f"microsoft/{type}")
        self.head = nn.Sequential(
            nn.AvgPool2d(7), nn.Flatten(), nn.Linear(2048, num_digits * num_chars)
        )
        self.num_digits = num_digits
        self.num_chars = num_chars

    def forward(self, x):
        inputs = self.image_processor(
            x, return_tensors="pt", do_rescale=False, device="cuda"
        ).to(x.device)
        outputs = self.model(**inputs)
        last_hidden_states = outputs.last_hidden_state
        flattened_output = self.head(last_hidden_states)
        return torch.reshape(
            flattened_output,
            [flattened_output.shape[0], self.num_chars, self.num_digits],
        )


def train_experiment(config=None):
    # Constants
    EXPERIMENT_NAME = "" + datetime.now().strftime("%Y-%m-%dT%H-%M-%S") + "_resnet"

    with wandb.init(config=config, name=EXPERIMENT_NAME, project="comp430-project"):
        config = wandb.config

        if config["printing"]:
            print("-" * 10 + "~TRAIN~" + "-" * 10)
        torch.manual_seed(config["rng_seed"])
        np.random.seed(config["rng_seed"])
        if config["printing"]:
            print("Loading datasets...")

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {device}")

        transforms = T.Compose([T.Resize(config["dataset"]["resize_hw"]), T.ToTensor()])

        if config["dataset"]["sample"] == "captcha-dataset":
            raw_dataset = CaptchaDataset(
                "parsasam/captcha-dataset", transform=transforms
            )
        elif config["dataset"]["sample"] == "large-captcha-dataset":
            raw_dataset = CaptchaDataset(
                "akashguna/large-captcha-dataset", transform=transforms
            )
        elif config["dataset"]["sample"] == "combined":
            raw_dataset = get_combined_dataset(transforms)
        else:
            raise AttributeError(
                f"Invalid dataset sample: {config['dataset']['sample']}"
            )

        train_set, val_set = random_split(raw_dataset, [0.8, 0.2])

        assert len(train_set) > len(
            val_set
        ), f"Sanity check failed, size of train set ({len(train_set)}) must be greater than size of val set ({len(val_set)})"

        train_loader = DataLoader(
            train_set,
            batch_size=config["dataset"]["batch_size"],
            shuffle=True,
            num_workers=config["dataset"]["num_workers"],
        )
        val_loader = DataLoader(
            val_set,
            batch_size=config["dataset"]["batch_size"],
            shuffle=True,
            num_workers=config["dataset"]["num_workers"],
        )

        if config["printing"]:
            print("Setting up model, optim, etc...")

        model = ResNetWrapper(
            num_digits=config["model"]["num_digits"],
            num_chars=config["model"]["num_chars"],
            type=config["model"]["type"],
        )
        loss_fn = torch.nn.CrossEntropyLoss()

        if config["optimizer"]["type"] == "rmsprop":
            optimizer = optim.RMSprop(
                model.parameters(), lr=config["optimizer"]["learning_rate"]
            )
        elif config["optimizer"]["type"] == "adam":
            optimizer = optim.RMSprop(
                model.parameters(), lr=config["optimizer"]["learning_rate"]
            )
        else:
            raise AttributeError(
                f"incorrect value for optimizer type: {config['optimizer']['type']}"
            )
        train(
            model=model,
            optimizer=optimizer,
            loss_fn=loss_fn,
            num_epoch=config["max_epochs"],
            train_loader=train_loader,
            val_loader=val_loader,
            experiment_name=EXPERIMENT_NAME,
            printing=True,
            tensorboard_logging=True,
            wandb_logging=True,
            metrics={
                "top-1-acc": torchmetrics.classification.MulticlassAccuracy(
                    num_classes=config["model"]["num_chars"]
                ),
                # "top-5-acc": torchmetrics.classification.MulticlassAccuracy(
                #     num_classes=config["model"]["num_chars"], top_k=5
                # ),
            },
            checkpointing=True,
            device=device,
        )


def _regular_training():
    train_experiment(
        config={
            "rng_seed": 42,
            "max_epochs": 50,
            "printing": True,
            "tensorboard_logging": True,
            "wandb_logging": True,
            "dataset": {
                "sample": "combined",
                # "sample": "captcha-dataset",
                # "sample": "large-captcha-dataset",
                "resize_hw": [224, 224],
                "batch_size": 64,
                "num_workers": 2,
            },
            "model": {
                "type": "resnet-101",
                "num_digits": 5,
                "num_chars": 62,
            },
            "optimizer": {
                "type": "adam",
                "learning_rate": 1e-4,
            },
        }
    )


# def _sweep():
#     sweep_id = wandb.sweep(sweep_config, project="sam2-replication")
#     wandb.agent(sweep_id, train_experiment, count=6)


if __name__ == "__main__":
    # _sweep()
    _regular_training()
    # _eval()
