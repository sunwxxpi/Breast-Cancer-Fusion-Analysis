from PIL import Image
import numpy as np
import os

def load_and_binarize_image(image_path):
    # 이미지를 경로에서 불러옵니다.
    image = Image.open(image_path).convert('L')  # 이미지를 그레이스케일로 변환
    # 이미지를 numpy 배열로 변환합니다.
    label_image = np.array(image)
    # 255는 1로, 나머지는 그대로 둡니다.
    binarized_image = np.where(label_image == 255, 1, 0).astype(np.uint8)
    return binarized_image

def save_binarized_image(binarized_image, save_path):
    # numpy 배열을 이미지로 변환합니다. 0과 1을 그대로 저장합니다.
    binarized_image_pil = Image.fromarray(binarized_image)
    # 이미지를 저장합니다.
    binarized_image_pil.save(save_path)

# 원본 라벨 이미지들이 있는 디렉토리 경로
original_labels_dir = '/home/psw/dataset/TestSet/labelsTr'
# 이진화된 라벨 이미지를 저장할 디렉토리 경로
binarized_labels_dir = '/home/psw/dataset/TestSet/normalized'

# 저장할 디렉토리가 존재하지 않으면 생성합니다.
if not os.path.exists(binarized_labels_dir):
    os.makedirs(binarized_labels_dir)

# 디렉토리 내의 모든 파일을 처리합니다.
for filename in os.listdir(original_labels_dir):
    if filename.endswith('.png'):
        original_file_path = os.path.join(original_labels_dir, filename)
        save_file_path = os.path.join(binarized_labels_dir, filename)
        binarized_image = load_and_binarize_image(original_file_path)
        save_binarized_image(binarized_image, save_file_path)

print("All label images have been binarized and saved to:", binarized_labels_dir)