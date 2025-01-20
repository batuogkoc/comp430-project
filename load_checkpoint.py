import torch
from torch.utils.data import DataLoader
import torchvision.transforms as T
from train import ResNetWrapper
from captcha_datasets.datasets import CaptchaDataset


def load_checkpoint(path):
    checkpoint = torch.load(path)

    print(checkpoint.keys())

    model = ResNetWrapper(5, 62, "resnet-101")
    model.load_state_dict(checkpoint["model_state_dict"])

    dataset = CaptchaDataset(
        "akashguna/large-captcha-dataset",
        transform=T.Compose((T.Resize((224, 224)), T.ToTensor())),
    )
    loader = DataLoader(dataset, 1, shuffle=False)

    loss_fn = torch.nn.CrossEntropyLoss()
    return model, loader, loss_fn


if __name__ == "__main__":
    PATH = "cleaned_checkpoint.pt"

    model, loader, loss_fn = load_checkpoint(PATH)

    # user code
    x, y = next(iter(loader))
    x.set_requires_grad(True)
    y_pred = model(x)
    loss = loss_fn(y_pred, y)
    loss.backward()
    x_grad = x.grad

    x_perturbed = x - 0.1 * x.grad

    y_perturbed_pred = model(x_perturbed)

    print(y, torch.argmax(y_pred, 1), torch.argmax(y_perturbed_pred, 1))
