# image_processor.py
from PIL import Image
import os

def convert_to_grayscale(image_path):
    try:
        with Image.open(image_path) as img:
            grayscale = img.convert("L")  # Convert to grayscale
            output_path = os.path.splitext(image_path)[0] + "_grayscale.jpeg"
            grayscale.save(output_path)
            return output_path
    except Exception as e:
        return str(e)
