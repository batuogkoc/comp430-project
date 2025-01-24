import os
import numpy as np
import random
from PIL import Image, ImageDraw, ImageFilter
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


def apply_random_patch(image):
    image_with_patch = image.copy()
    draw = ImageDraw.Draw(image_with_patch)

    num_patches = np.random.randint(1, 6) 

    for _ in range(num_patches):
        patch_ratio = np.random.uniform(0.15, 0.3) 
        patch_size = int(min(image.width, image.height) * patch_ratio)
        patch_size = max(10, patch_size) 

        patch_color = tuple(np.random.randint(0, 256, size=3)) 
        alpha = np.random.randint(50, 200)  

        center_x = image.width // 2
        center_y = image.height // 2

        max_offset_x = max(1, (image.width - patch_size) // 2) 
        max_offset_y = max(1, (image.height - patch_size) // 2)

        top_left_x = np.random.randint(center_x - max_offset_x, center_x + max_offset_x - patch_size + 1)
        top_left_y = np.random.randint(center_y - max_offset_y, center_y + max_offset_y - patch_size + 1)

        shape_type = np.random.choice(["rectangle", "circle", "ellipse", "polygon", "line", "star"])

        overlay = Image.new("RGBA", image.size, (0, 0, 0, 0))
        overlay_draw = ImageDraw.Draw(overlay)

        if shape_type == "rectangle":
            bottom_right_x = top_left_x + patch_size
            bottom_right_y = top_left_y + patch_size
            overlay_draw.rectangle(
                [top_left_x, top_left_y, bottom_right_x, bottom_right_y],
                fill=patch_color + (alpha,)
            )
        elif shape_type == "circle":
            radius = patch_size // 2
            overlay_draw.ellipse(
                [top_left_x, top_left_y, top_left_x + 2 * radius, top_left_y + 2 * radius],
                fill=patch_color + (alpha,)
            )
        elif shape_type == "ellipse":
            overlay_draw.ellipse(
                [top_left_x, top_left_y, top_left_x + patch_size, top_left_y + patch_size // 2],
                fill=patch_color + (alpha,)
            )
        elif shape_type == "polygon":
            num_vertices = np.random.randint(3, 6)  
            vertices = [
                (np.random.randint(top_left_x, top_left_x + patch_size),
                 np.random.randint(top_left_y, top_left_y + patch_size))
                for _ in range(num_vertices)
            ]
            overlay_draw.polygon(vertices, fill=patch_color + (alpha,))
        elif shape_type == "line":
            end_x = np.random.randint(top_left_x, top_left_x + patch_size)
            end_y = np.random.randint(top_left_y, top_left_y + patch_size)
            overlay_draw.line(
                [top_left_x, top_left_y, end_x, end_y],
                fill=patch_color + (alpha,), width=np.random.randint(1, 5)
            )
        elif shape_type == "star":
            num_points = 5
            angle = 2 * np.pi / num_points
            center_x = top_left_x + patch_size // 2
            center_y = top_left_y + patch_size // 2
            radius_outer = patch_size // 2
            radius_inner = patch_size // 4
            points = []
            for i in range(num_points * 2):
                r = radius_outer if i % 2 == 0 else radius_inner
                theta = i * angle / 2
                x = center_x + r * np.cos(theta)
                y = center_y + r * np.sin(theta)
                points.append((x, y))
            overlay_draw.polygon(points, fill=patch_color + (alpha,))

        image_with_patch = Image.alpha_composite(image_with_patch.convert("RGBA"), overlay).convert("RGB")

    blur_radius = np.random.uniform(1.0, 1.20)
    image_with_patch = image_with_patch.filter(ImageFilter.GaussianBlur(blur_radius))

    return image_with_patch


def create_perturbed_images(input_folder, output_folder, max_images=100):

    os.makedirs(output_folder, exist_ok=True)

    image_files = [f for f in os.listdir(input_folder) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff'))]
    random.shuffle(image_files) 
    image_files = image_files[:max_images] 

    for filename in image_files:
        input_path = os.path.join(input_folder, filename)

        try:
            image = Image.open(input_path).convert("RGB")

            perturbed_image = apply_random_patch(image)

            output_path = os.path.join(output_folder, filename)
            perturbed_image.save(output_path)

            print(f"Perturbed image saved: {output_path}")
        except Exception as e:
            print(f"Error processing {filename}: {e}")

def visualize(model, loader, num, device):
    image_files = [
        f for f in os.listdir("/Users/idilgorgulu/Desktop/test_sample_for_patch")
        if f.lower().endswith((".png", ".jpg", ".jpeg", ".bmp", ".tiff"))
    ]
    image_files = [os.path.join("/Users/idilgorgulu/Desktop/test_sample_for_patch", f) for f in image_files]

    model.eval()
    model.to(device)
    num = 3 
    fig, axs = plt.subplots(num, 2, figsize=(10, num * 3))
    fig.subplots_adjust(wspace=0.2, hspace=0.3)

    for i, (x, y) in enumerate(loader):
        if i >= num:
            break

        image_path1 = random.choice(image_files)
        image1 = Image.open(image_path1)
        x_perturbed1 = apply_random_patch(image1)  
        x_perturbed1 = x_perturbed1.convert("RGB")
        x_perturbed1 = transforms(x_perturbed1).unsqueeze(0).to(device)

        y_perturbed_pred1 = model(x_perturbed1)
        y_perturbed_pred_str1 = digits_to_label(torch.argmax(y_perturbed_pred1, 1)[0])  

        axs[i, 0].imshow(torch.permute(x_perturbed1[0], (1, 2, 0)).detach().cpu().numpy())
        axs[i, 0].set_title(f"P: {y_perturbed_pred_str1}", fontsize=10)
        axs[i, 0].axis("off")  

        image_path2 = random.choice(image_files)
        image2 = Image.open(image_path2)
        x_perturbed2 = apply_random_patch(image2) 
        x_perturbed2 = x_perturbed2.convert("RGB")
        x_perturbed2 = transforms(x_perturbed2).unsqueeze(0).to(device)

        y_perturbed_pred2 = model(x_perturbed2)
        y_perturbed_pred_str2 = digits_to_label(torch.argmax(y_perturbed_pred2, 1)[0])
        axs[i, 1].imshow(torch.permute(x_perturbed2[0], (1, 2, 0)).detach().cpu().numpy())
        axs[i, 1].set_title(f"P: {y_perturbed_pred_str2}", fontsize=10)
        axs[i, 1].axis("off")  # Turn off axes

    fig.savefig("visualization_patch_new.png")

if __name__ == "__main__":
    input_folder = "/Users/idilgorgulu/Desktop/test_sample_for_patch" 
    output_folder = "perturbed_images/patch" 
    create_perturbed_images(input_folder, output_folder, max_images=100)
    DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Using device: {DEVICE}")

    CHECKPOINT_PATH = "/Users/idilgorgulu/Desktop/cleaned_checkpoint_1.pt"
    model, loss_fn = load_checkpoint(CHECKPOINT_PATH, DEVICE)
    transforms = T.Compose(
        (
            T.Resize((224, 224)),
            T.ToTensor(),
        )
    ) 
    (train_set, val_set, test_set), _ = split_dataset(
        get_combined_dataset(transforms),
        [0.8, 0.1, 0.1],
        "combined_splits_indices.cache",
    )
    loader = DataLoader(test_set, batch_size=1, shuffle=True)

    num_samples_to_visualize = 5 
    visualize(model, loader, num_samples_to_visualize, DEVICE)
