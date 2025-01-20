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


def peek_inside_model(model, x):
    # inputs = model.image_processor(
    #     x, return_tensors="pt", do_rescale=False, device="cuda"
    # ).to(x.device)
    inputs = {
        "pixel_values": T.Normalize(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
        )(x)
    }
    # print(torch.allclose)
    # print(dir(model.image_processor))
    outputs = model.model(**inputs)
    # outputs = model.model(pixel_values=x)
    last_hidden_states = outputs.last_hidden_state
    flattened_output = model.head(last_hidden_states)
    return torch.reshape(
        flattened_output,
        [flattened_output.shape[0], model.num_chars, model.num_digits],
    )


def fast_gradient_sign(model, x, y):
    x_perturbed = x.clone().detach().requires_grad_(True)
    y_pred = peek_inside_model(model, x_perturbed)
    loss = loss_fn(y_pred, y)
    model.zero_grad()
    loss.backward()
    x_perturbed = x_perturbed + 0.1 * torch.sign(x_perturbed.grad)
    return torch.clamp(x_perturbed, 0, 1)


if __name__ == "__main__":
    PATH = "cleaned_checkpoint.pt"

    model, loss_fn = load_checkpoint(PATH)

    # user code
    dataset = CaptchaDataset(
        "akashguna/large-captcha-dataset",
        transform=T.Compose(
            (
                T.Resize((224, 224)),
                T.ToTensor(),
                # T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            )
        ),
    )
    loader = DataLoader(dataset, batch_size=1, shuffle=True)
    x, y = next(iter(loader))
    x_perturbed = x.clone().detach().requires_grad_(True)
    # y_pred = model(x)
    y_target = y
    # y_target = dataset.label_to_digits("aaaaa").unsqueeze(0)
    y_pred = peek_inside_model(model, x_perturbed)
    # for i in range(10):
    #     loss = loss_fn(peek_inside_model(model, x_perturbed), y_target)
    #     model.zero_grad()
    #     loss.backward()
    #     print(loss)
    #     print(x_perturbed.requires_grad)
    #     # x_perturbed = x_perturbed + 0.05 * torch.sign(x_perturbed.grad)
    #     x_perturbed = x_perturbed - 0.04 * x_perturbed.grad
    # x_perturbed = torch.clamp(x_perturbed, 0, 1)
    x_perturbed = fast_gradient_sign(model, x_perturbed, y)
    y_perturbed_pred = model(x_perturbed)

    y_str = digits_to_label(y[0])
    y_target_str = digits_to_label(y_target[0])
    y_pred_str = digits_to_label(torch.argmax(y_pred, 1)[0])
    y_perturbed_pred_str = digits_to_label(torch.argmax(y_perturbed_pred, 1)[0])
    print("gt:", y_str)
    print("pred:", y_pred_str)
    print("perturbed:", y_perturbed_pred_str)
    print("target:", y_target_str)
    fig, axs = plt.subplots(1, 2)
    axs[0].set_title(f"original: {y_pred_str}")
    axs[0].imshow(torch.permute(x[0], (1, 2, 0)).detach().cpu().numpy())
    axs[0].axis("off")
    axs[1].set_title(f"{y_target_str}: {y_perturbed_pred_str}")
    axs[1].imshow(torch.permute(x_perturbed[0], (1, 2, 0)).detach().cpu().numpy())
    axs[1].axis("off")

    fig.savefig("image")
