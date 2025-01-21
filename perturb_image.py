import os
import numpy as np
from PIL import Image, ImageDraw, ImageFilter

def apply_random_patch(image):

    image_with_patch = image.copy()
    draw = ImageDraw.Draw(image_with_patch)

    num_patches = np.random.randint(1, 6) 

    for _ in range(num_patches):
        patch_ratio = np.random.uniform(0.1, 0.3) 
        patch_size = int(min(image.width, image.height) * patch_ratio)
        patch_size = max(1, patch_size)  

        patch_color = tuple(np.random.randint(0, 256, size=3)) 

        alpha = np.random.randint(50, 200)  

        max_x = image.width - patch_size
        max_y = image.height - patch_size

        top_left_x = np.random.randint(0, max_x + 1)
        top_left_y = np.random.randint(0, max_y + 1)

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

    blur_radius = np.random.uniform(1.5, 1.5)  
    image_with_patch = image_with_patch.filter(ImageFilter.GaussianBlur(blur_radius))

    return image_with_patch

def create_perturbed_images(input_folder, output_folder, max_images=100):

    os.makedirs(output_folder, exist_ok=True)

    image_files = [f for f in os.listdir(input_folder) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff'))]
    image_files = sorted(image_files)[:max_images]  

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

if __name__ == "__main__":
    input_folder = "archive" 
    output_folder = "perturbed_images" 

    create_perturbed_images(input_folder, output_folder, max_images=100)
