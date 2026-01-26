import cv2
import os

def pad_to_square(image, padding_color=127):
    height, width = image.shape[:2]
    if height == width:
        return image
    max_side = max(height, width)
    top = (max_side - height) // 2
    bottom = max_side - height - top
    left = (max_side - width) // 2
    right = max_side - width - left
    padded_image = cv2.copyMakeBorder(image, top, bottom, left, right, cv2.BORDER_CONSTANT, value=[padding_color, padding_color, padding_color])
    return padded_image

def process_images(input_dir, output_dir, target_size):
    # Create output directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Iterate through all files and subdirectories in the input directory
    for root, _, files in os.walk(input_dir):
        for filename in files:
            if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff')):
                # Construct full file path
                file_path = os.path.join(root, filename)
                
                # Read the image in color
                image = cv2.imread(file_path, cv2.IMREAD_COLOR)
                if image is None:
                    continue

                # Pad the image to make it square with gray padding
                square_image = pad_to_square(image, padding_color=0)
                
                # Resize the image to the target size
                resized_image = cv2.resize(image, (target_size, target_size))
                
                # Construct the output file path
                relative_path = os.path.relpath(root, input_dir)
                output_file_dir = os.path.join(output_dir, relative_path)
                
                if not os.path.exists(output_file_dir):
                    os.makedirs(output_file_dir)
                
                output_file_path = os.path.join(output_file_dir, filename)
                
                # Save the processed image
                cv2.imwrite(output_file_path, resized_image)
                print(f"Processed and saved: {output_file_path}")

# Example usage
input_directory = './data/train/elastogram_gray'
output_directory = './data/train/elastogram_gray_square'
target_size = 352
process_images(input_directory, output_directory, target_size)