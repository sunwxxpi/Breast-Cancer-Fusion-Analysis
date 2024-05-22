import cv2
import numpy as np
import os
import glob

def crop_strain_elastography(image):
    lower_black = np.array([0, 0, 0])
    upper_black = np.array([15, 15, 15])
    mask = cv2.inRange(image, lower_black, upper_black)
    
    mask = cv2.bitwise_not(mask)
    image = cv2.bitwise_and(image, image, mask=mask)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if contours:
        largest_contour = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(largest_contour)
        
        x = x + 15
        y = y + 15
        w = w - 30
        h = h - 30
        cropped_image = image[y:y+h, x:x+w]
        
    return x, y, w, h, cropped_image


def crop_elastogram(image):
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    lower_bound = np.array([0, 50, 50])
    upper_bound = np.array([179, 255, 255])
    color_mask = cv2.inRange(hsv_image, lower_bound, upper_bound)

    contours, _ = cv2.findContours(color_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if contours:
        largest_contour = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(largest_contour)
        
        x = x + 15
        y = y + 15
        w = w - 30
        h = h - 30
        cropped_image = image[y:y+h, x:x+w]

    return x, y, w, h, cropped_image


def crop_elastogram_from_grayscale(image, sr_x, sr_y, brr_x, brr_y, brr_w, brr_h):
    target_x = sr_x + brr_x
    target_y = sr_y + brr_y
    target_w = brr_w
    target_h = brr_h
    
    cropped_image = image[target_y:target_y+target_h, target_x-722:target_x-722+target_w]
    
    return cropped_image


def find_rightmost_rgb_column(image):
    color_mask = (image[:,:,0] != image[:,:,1]) | (image[:,:,1] != image[:,:,2]) | (image[:,:,0] != image[:,:,2])

    width = image.shape[1]

    for x in range(width-1, -1, -1):
        if np.any(color_mask[:, x]):
            return x

    return -1


def find_bottommost_gray_row(image):
    gray_mask = (image[:,:,0] == image[:,:,1]) & (image[:,:,1] == image[:,:,2])

    height = image.shape[0]

    for y in range(height-1, -1, -1):
        if np.any(gray_mask[y, :]):
            return y

    return -1


def remove_left_rgb_part(image, offset_x):
    if offset_x is not None and offset_x > 0:
        return image[:, offset_x+1:]
    
    return image


def remove_top_gray_part(image, offset_y):
    if offset_y is not None and offset_y > 0:
        return image[offset_y+1:, :]
    
    return image


def ensure_dir(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)


def extract_elastogram(base_dir, output_dir):
    for score in range(1, 6):
        input_path = f"{base_dir}/score{score}/*.png"
        images = glob.glob(input_path)
        
        for image_path in images:
            try:
                output_subdir = os.path.join(output_dir, f"elastogram_cropped/score{score}")
                ensure_dir(output_subdir)
                
                grayscale_output_subdir = os.path.join(output_dir, f"elastogram_from_grayscale_cropped/score{score}")
                ensure_dir(grayscale_output_subdir)
                
                file_name = os.path.basename(image_path)
                output_path = os.path.join(output_subdir, file_name)
                grayscale_output_path = os.path.join(grayscale_output_subdir, file_name)
                
                image = cv2.imread(image_path)
                
                sr_x, sr_y, _, _, strain_region = crop_strain_elastography(image)
                
                brr_x, brr_y, brr_w, brr_h, elastogram = crop_elastogram(strain_region)
                elastogram_from_grayscale = crop_elastogram_from_grayscale(image, sr_x, sr_y, brr_x, brr_y, brr_w, brr_h)
                
                offset_x = find_rightmost_rgb_column(elastogram_from_grayscale)
                elastogram_cropped = remove_left_rgb_part(elastogram, offset_x)
                elastogram_from_grayscale_cropped = remove_left_rgb_part(elastogram_from_grayscale, offset_x)
                
                offset_y = find_bottommost_gray_row(elastogram_cropped)
                elastogram_cropped = remove_top_gray_part(elastogram_cropped, offset_y)
                elastogram_from_grayscale_cropped = remove_top_gray_part(elastogram_from_grayscale_cropped, offset_y)                

                cv2.imwrite(output_path, elastogram_cropped)
                print(f"Saved {output_path}")
                
                cv2.imwrite(grayscale_output_path, elastogram_from_grayscale_cropped)
                print(f"Saved {grayscale_output_path}")
                
            except Exception as e:
                print(f"Error processing {image_path}: {e}")
            

base_dir = './strain_png/S'
output_dir = './strain_png'
extract_elastogram(base_dir, output_dir)