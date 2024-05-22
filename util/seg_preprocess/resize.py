import os
import cv2

# 이미지와 레이블의 크기를 맞추는 함수
def resize_images_and_labels(image_path, label_path, output_image_path, output_label_path):
    image = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
    label = cv2.imread(label_path, cv2.IMREAD_UNCHANGED)

    # 이미지와 레이블 크기 가져오기
    image_shape = image.shape
    label_shape = label.shape

    # 크기 맞추기
    if image_shape != label_shape:
        target_shape = label_shape[:2]  # (height, width)
        resized_image = cv2.resize(image, (target_shape[1], target_shape[0]), interpolation=cv2.INTER_NEAREST)
        
        # 이미지 저장
        cv2.imwrite(output_image_path, resized_image)
        cv2.imwrite(output_label_path, label)
    else:
        # 이미지 저장 (크기가 동일한 경우)
        cv2.imwrite(output_image_path, image)
        cv2.imwrite(output_label_path, label)

# 원본 디렉토리 경로 설정
base_dir = '/home/psw/dataset/B(U-Mamba)'
image_dir = os.path.join(base_dir, 'imagesTr')
label_dir = os.path.join(base_dir, 'labelsTr')

# 출력 디렉토리 경로 설정
output_base_dir = '/home/psw/dataset/B(U-Mamba)_resized'
output_image_dir = os.path.join(output_base_dir, 'imagesTr')
output_label_dir = os.path.join(output_base_dir, 'labelsTr')

# 출력 디렉토리 생성
os.makedirs(output_image_dir, exist_ok=True)
os.makedirs(output_label_dir, exist_ok=True)

# 파일 목록 가져오기
image_files = sorted([f for f in os.listdir(image_dir) if f.endswith('.png')])
label_files = sorted([f for f in os.listdir(label_dir) if f.endswith('.png')])

# 이미지와 레이블 크기 조정 및 저장
for image_file, label_file in zip(image_files, label_files):
    image_path = os.path.join(image_dir, image_file)
    label_path = os.path.join(label_dir, label_file)
    output_image_path = os.path.join(output_image_dir, image_file)
    output_label_path = os.path.join(output_label_dir, label_file)

    resize_images_and_labels(image_path, label_path, output_image_path, output_label_path)

print("이미지와 레이블의 크기 조정이 완료되었습니다.")