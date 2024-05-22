from PIL import Image
import numpy as np
import os

def load_binarized_image(image_path):
    # 이미지를 경로에서 불러옵니다.
    image = Image.open(image_path).convert('L')  # 이미지를 그레이스케일로 변환
    # 이미지를 numpy 배열로 변환합니다.
    binarized_image = np.array(image) # 이진화된 이미지를 불러옴
    return binarized_image

def denormalize_image(binarized_image):
    # 이진화된 이미지를 다시 0과 255로 변환합니다.
    denormalized_image = binarized_image * 255
    return denormalized_image

def save_denormalized_image(denormalized_image, save_path):
    # numpy 배열을 이미지로 변환합니다.
    denormalized_image_pil = Image.fromarray(denormalized_image.astype(np.uint8))
    # 이미지를 저장합니다.
    denormalized_image_pil.save(save_path)

# 이진화된 라벨 이미지들이 있는 디렉토리 경로
binarized_labels_dir = '/home/psw/dataset/dir1'
# denormalized 이미지를 저장할 디렉토리 경로
denormalized_labels_dir = '/home/psw/dataset/dir2'

# 저장할 디렉토리가 존재하지 않으면 생성합니다.
if not os.path.exists(denormalized_labels_dir):
    os.makedirs(denormalized_labels_dir)

# 디렉토리 내의 모든 파일을 처리합니다.
for filename in os.listdir(binarized_labels_dir):
    if filename.endswith('.png'):
        binarized_file_path = os.path.join(binarized_labels_dir, filename)
        denormalized_save_file_path = os.path.join(denormalized_labels_dir, filename)
        
        # 이진화된 이미지를 로드합니다.
        binarized_image = load_binarized_image(binarized_file_path)
        # 이미지를 denormalize합니다.
        denormalized_image = denormalize_image(binarized_image)
        # denormalize된 이미지를 저장합니다.
        save_denormalized_image(denormalized_image, denormalized_save_file_path)

print("All binarized label images have been denormalized and saved to:", denormalized_labels_dir)