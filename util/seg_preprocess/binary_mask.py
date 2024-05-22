import os

from tqdm import tqdm
import cv2

def convert_images_to_binary(source_directories, output_directory):
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)
    
    for source_dir in source_directories:
        labelsTr_dir = os.path.join(source_dir, 'labelsTr')
        if not os.path.exists(labelsTr_dir):
            print(f"Directory {labelsTr_dir} does not exist. Skipping...")
            continue
        
        # Get list of image files in the labelsTr directory
        image_files = [f for f in os.listdir(labelsTr_dir) if os.path.isfile(os.path.join(labelsTr_dir, f))]
        
        # Progress bar for the current source directory
        for filename in tqdm(image_files, desc=f"Processing {source_dir}"):
            file_path = os.path.join(labelsTr_dir, filename)
            
            # Read the image in grayscale mode
            image = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
            if image is None:
                print(f"Failed to read {file_path}. Skipping...")
                continue
            
            # Apply threshold to binarize the image
            _, binary_image = cv2.threshold(image, 127, 255, cv2.THRESH_BINARY)
            
            # Create corresponding output directory structure
            output_subdir = os.path.join(output_directory, os.path.relpath(labelsTr_dir, './'))
            if not os.path.exists(output_subdir):
                os.makedirs(output_subdir)
            
            # Save the binary image to the output directory
            output_file_path = os.path.join(output_subdir, filename)
            cv2.imwrite(output_file_path, binary_image)

source_directories = ['./BUSI', './STU', './TDSC-ABUS2023', './TestSet']
output_directory = './B(binary_converted)'

convert_images_to_binary(source_directories, output_directory)