import torch
from torch.utils.data import DataLoader
import torchvision.transforms as T
from train import ResNetWrapper
from captcha_datasets.datasets import CaptchaDataset, index_to_char, digits_to_label
from matplotlib import pyplot as plt


def load_checkpoint(path):
    checkpoint = torch.load(path)

    print(checkpoint.keys())

    model = ResNetWrapper(5, 62, "resnet-101")
    model.load_state_dict(checkpoint["model_state_dict"])

    loss_fn = torch.nn.CrossEntropyLoss()
    return model, loss_fn


if __name__ == "__main__":
    PATH = "cleaned_checkpoint.pt"

    model, loss_fn = load_checkpoint(PATH)

    # user code
    dataset = CaptchaDataset(
        "akashguna/large-captcha-dataset",
        transform=T.Compose((T.Resize((224, 224)), T.ToTensor())),
    )
    loader = DataLoader(dataset, batch_size=1, shuffle=True)
    x, y = next(iter(loader))
    x = torch.tensor(x, requires_grad=True)
    print(x.requires_grad)
    y_pred = model(x)
    loss = loss_fn(y_pred, y)
    model.zero_grad()
    loss.backward()
    print(x.grad)
    print(loss)
    print(digits_to_label(torch.argmax(y_pred, 1)[0]))
    print(digits_to_label(y[0]))
    plt.imshow(torch.permute(x[0], (1, 2, 0)).detach().cpu().numpy())
    plt.savefig("image")
    # x_perturbed = x - 0.1 * x.grad

    # y_perturbed_pred = model(x_perturbed)

    # print(y, torch.argmax(y_pred, 1), torch.argmax(y_perturbed_pred, 1))
