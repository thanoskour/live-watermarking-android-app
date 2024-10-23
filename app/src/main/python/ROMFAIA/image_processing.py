from PIL import Image, ImageDraw, ImageFont
import os

def add_watermark_pillow(image_path, watermark_text):
    # Load the image
    img = Image.open(image_path)
    draw = ImageDraw.Draw(img)

    # Set the position and font for the watermark
    position = (50, 50)  # Change as needed
    font = ImageFont.load_default()  # Or specify a font file

    # Add the watermark
    draw.text(position, watermark_text, (255, 255, 255))  # White text

    # Save the watermarked image
    output_path  = os.path.splitext(image_path)[0] + "_watermark.jpeg"
    img.save(output_path)

    return output_path
