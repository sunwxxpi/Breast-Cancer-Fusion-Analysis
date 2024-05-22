import os

from tqdm import tqdm
from PIL import Image

def convert_images_to_grayscale(source_directories, output_directory):
    image_paths = []
    for source_directory in source_directories:
        for subdir, dirs, files in os.walk(source_directory):
            for filename in files:
                if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                    image_paths.append((subdir, filename))

    for subdir, filename in tqdm(image_paths, desc="Converting images"):
        img_path = os.path.join(subdir, filename)
        rel_path = os.path.relpath(subdir, source_directory)
        output_subdir = os.path.join(output_directory, os.path.basename(source_directory), rel_path)
        if not os.path.exists(output_subdir):
            os.makedirs(output_subdir)

        with Image.open(img_path) as img:
            grayscale_img = img.convert('L')
            output_path = os.path.join(output_subdir, filename)
            grayscale_img.save(output_path)


source_directories = ['./BUSI', './TDSC-ABUS2023', './TestSet']
output_directory = './B(gray_converted)'

convert_images_to_grayscale(source_directories, output_directory)