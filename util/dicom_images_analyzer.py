import os
import pydicom
import numpy as np

def analyze_dicom_files(directory, output_file):
    with open(output_file, "w") as file:
        for filename in os.listdir(directory):
            file_path = os.path.join(directory, filename)
            
            if file_path.lower().endswith('.dcm'):
                try:
                    ds = pydicom.dcmread(file_path, force=True)
                    
                    pixel_array = ds.pixel_array
                    data_type = pixel_array.dtype
                    min_pixel = np.min(pixel_array)
                    max_pixel = np.max(pixel_array)
                    
                    result = (f"File: {filename}\n"
                              f"  Dimensions: {pixel_array.shape}\n"
                              f"  Data Type: {data_type}\n"
                              f"  Pixel Range: {min_pixel} to {max_pixel}\n\n")
                    
                    print(result)
                    file.write(result)
                    
                except Exception as e:
                    error_msg = f"Failed to process {filename}: {e}\n\n"
                    
                    print(error_msg)
                    file.write(error_msg)


directory = './strain_dcm/score1'
output_file = './strain_dcm/socre1_analysis.txt'
analyze_dicom_files(directory, output_file)