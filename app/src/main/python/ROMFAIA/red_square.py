import cv2
import os

def add_red_square_to_image(image_path):
    # Load the image
    image = cv2.imread(image_path)

    if image is None:
        return "Image not found"

    # Define the size and position of the square
    square_size = 50
    top_left_corner = (20, 20)  # Modify this to place the square wherever you want
    bottom_right_corner = (top_left_corner[0] + square_size, top_left_corner[1] + square_size)

    # Draw a red square (Color in BGR format)
    red_color = (0, 0, 255)
    thickness = -1  # Negative thickness for a filled square
    cv2.rectangle(image, top_left_corner, bottom_right_corner, red_color, thickness)

    # Save the modified image
    output_path = os.path.splitext(image_path)[0] + "_red_square.jpg"
    cv2.imwrite(output_path, image)

    return output_path
