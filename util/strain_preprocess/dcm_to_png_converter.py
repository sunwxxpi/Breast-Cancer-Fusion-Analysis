import os
import pydicom
from PIL import Image

def convert_dicom_to_png(dicom_file_path, output_folder):
    try:
        ds = pydicom.dcmread(dicom_file_path, force=True)
        
        if not hasattr(ds, 'file_meta') or 'TransferSyntaxUID' not in ds.file_meta:
            if not hasattr(ds, 'file_meta'):
                ds.file_meta = pydicom.dataset.FileMetaDataset()
            ds.file_meta.TransferSyntaxUID = pydicom.uid.ExplicitVRLittleEndian
            
        if not hasattr(ds, 'PixelData'):
            print(f"No Pixel Data in {dicom_file_path}")
            return

        pixel_array = ds.pixel_array
        mode = 'L'

        if pixel_array.ndim == 2:
            mode = 'L'
        elif pixel_array.ndim == 3 and pixel_array.shape[2] == 3:
            mode = 'RGB'
        else:
            raise ValueError(f"Unsupported number of dimensions or color channels in {dicom_file_path}")

        image = Image.fromarray(pixel_array, mode)
        base_name = os.path.basename(dicom_file_path)
        output_path = os.path.join(output_folder, base_name.replace('.dcm', '.png'))
        
        os.makedirs(output_folder, exist_ok=True)
        image.save(output_path)
        print(f"Image saved at {output_path}")

    except Exception as e:
        print(f"Failed to process {dicom_file_path}: {e}")


def process_directory(input_base_directory, output_base_directory):
    for subdir in os.listdir(input_base_directory):
        input_subdir_path = os.path.join(input_base_directory, subdir)
        output_subdir_path = os.path.join(output_base_directory, subdir)

        if os.path.isdir(input_subdir_path):
            for filename in os.listdir(input_subdir_path):
                if filename.endswith('.dcm'):
                    dicom_file_path = os.path.join(input_subdir_path, filename)
                    convert_dicom_to_png(dicom_file_path, output_subdir_path)


input_base_directory = './strain_dcm'
output_base_directory = './strain_png'
process_directory(input_base_directory, output_base_directory)