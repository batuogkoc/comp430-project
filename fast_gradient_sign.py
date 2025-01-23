import torch
from torch.utils.data import DataLoader, Subset
import torchvision.transforms as T
from train import ResNetWrapper
from captcha_datasets.datasets import (
    CaptchaDataset,
    index_to_char,
    digits_to_label,
    get_combined_dataset,
    split_dataset,
)
from matplotlib import pyplot as plt
from evaluate_performance import load_checkpoint, evaluate_performance
from tqdm import tqdm
import numpy as np


def fast_gradient_sign(model, x, y, epsilon=0.1):
    x_perturbed = x.clone().detach().requires_grad_(True)
    y_pred = model(x_perturbed)
    loss = loss_fn(y_pred, y)
    model.zero_grad()
    loss.backward()
    x_perturbed = x_perturbed + epsilon * torch.sign(x_perturbed.grad)
    return torch.clamp(x_perturbed, 0, 1)


def visualize(model, loader, num, epsilons, device):
    model.eval()
    model.to(device)
    fig, axs = plt.subplots(num, len(epsilons) + 1, figsize=(10, 9.5))
    fig.subplots_adjust(wspace=0, hspace=0)
    axs[num - 1, 0].set_xlabel(f"Epsilon: {0}")
    for i, (x, y) in enumerate(loader):
        if i >= num:
            break
        x = x.to(device)
        y = y.to(device)
        x = x.clone().detach().requires_grad_(True)
        y_pred = model(x)
        y_str = digits_to_label(y[0])
        y_pred_str = digits_to_label(torch.argmax(y_pred, 1)[0])
        axs[i, 0].set_title(f"P: {y_pred_str}")
        axs[i, 0].set_ylabel(f"GT: {y_str}")
        axs[i, 0].imshow(torch.permute(x[0], (1, 2, 0)).detach().cpu().numpy())
        # axs[i, 0].axis("off")
        axs[i, 0].spines["top"].set_visible(False)
        axs[i, 0].spines["right"].set_visible(False)
        axs[i, 0].spines["left"].set_visible(False)
        axs[i, 0].spines["bottom"].set_visible(False)
        axs[i, 0].set_xticks([])  # Remove x-ticks
        axs[i, 0].set_yticks([])  # Remove y-ticks
        for j, epsilon in enumerate(epsilons):
            x_perturbed = fast_gradient_sign(model, x, y, epsilon=epsilon)
            y_perturbed_pred = model(x_perturbed)

            y_perturbed_pred_str = digits_to_label(torch.argmax(y_perturbed_pred, 1)[0])
            # print("gt:", y_str)
            # print("pred:", y_pred_str)
            # print("perturbed:", y_perturbed_pred_str)
            # print("target:", y_target_str)

            axs[i, j + 1].set_title(f"P: {y_perturbed_pred_str}")
            if i == num - 1:
                print("a")
                axs[i, j + 1].set_xlabel(f"Epsilon: {epsilon}")
            axs[i, j + 1].imshow(
                torch.permute(x_perturbed[0], (1, 2, 0)).detach().cpu().numpy()
            )
            # axs[i, j + 1].axis("off")
            axs[i, j + 1].spines["top"].set_visible(False)
            axs[i, j + 1].spines["right"].set_visible(False)
            axs[i, j + 1].spines["left"].set_visible(False)
            axs[i, j + 1].spines["bottom"].set_visible(False)
            axs[i, j + 1].set_xticks([])  # Remove x-ticks
            axs[i, j + 1].set_yticks([])  # Remove y-ticks

    fig.savefig("image")


def evaluate_fgsm(model, loader, epsilon, device):
    original_accuracy = 0
    perturbed_accuracy = 0
    n = 0
    model.eval()
    model.to(device)
    for x, y in tqdm(loader):
        x = x.to(device)
        y = y.to(device)
        x_perturbed = x.clone().detach().requires_grad_(True)
        y_pred = model(x_perturbed)

        x_perturbed = fast_gradient_sign(model, x_perturbed, y, epsilon=epsilon)
        y_perturbed_pred = model(x_perturbed)

        original_accuracy += torch.sum(
            torch.all((torch.argmax(y_pred, dim=1) == y), dim=-1).to(torch.float32)
        )
        perturbed_accuracy += torch.sum(
            torch.all((torch.argmax(y_perturbed_pred, dim=1) == y), dim=-1).to(
                torch.float32
            )
        )
        n += x.shape[0]

    return (original_accuracy / n).item(), (perturbed_accuracy / n).item()


def plot_epsilon_dependency(model, loader, device):
    epsilons = np.logspace(-3, -0.7, 10)
    accuracies = []
    for epsilon in epsilons:
        _, accuracy = evaluate_fgsm(model, loader, epsilon, device)
        accuracies.append(accuracy * 100)

    plt.xlabel("Epsilon")
    plt.ylabel("Accuracy (%)")
    plt.title("Accuracy vs Epsilon")
    plt.plot(epsilons, accuracies)
    plt.savefig("acc_vs_eps.png")


if __name__ == "__main__":
    PATH = "cleaned_checkpoint_1.pt"
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("using: ", DEVICE)

    model, loss_fn = load_checkpoint(PATH)

    transforms = T.Compose(
        (
            T.Resize((224, 224)),
            T.ToTensor(),
        )
    )
    # user code
    (train_set, val_set, test_set), _ = split_dataset(
        get_combined_dataset(transforms),
        [0.8, 0.1, 0.1],
        "combined_splits_indices.cache",
    )

    loader = DataLoader(test_set, batch_size=1, shuffle=True)
    visualize(model, loader, 4, [1e-2, 0.05, 0.1, 0.2], DEVICE)

    # test_subset = Subset(test_set, range(100))
    # loader = DataLoader(test_subset, batch_size=16, shuffle=True)

    # orig, pert = evaluate_fgsm(model, loader, 0.1, DEVICE)
    # print("orig: ", orig, "pert: ", pert)
    # plot_epsilon_dependency(model, loader, DEVICE)
