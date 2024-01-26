
import os
from PIL import Image


def load_images_from_directory(directory, target_size=(299, 299)):
    images = {}
    for filename in os.listdir(directory):
        if filename.lower().endswith((".jpg", ".png", ".jpeg")):
            img_path = os.path.join(directory, filename)
            with Image.open(img_path) as img:
                img = img.convert('RGB')  # Convert to RGB for model compatibility.
                img = img.resize(target_size, Image.Resampling.LANCZOS)  # Use high-quality downsampling filter.
                images[filename] = img  # Store the resized image.
    return images
