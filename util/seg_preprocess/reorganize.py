import json
import os
import shutil

# 파일 복사 함수
def copy_files(file_list, image_dest, label_dest, prefix, start_idx=1):
    idx = start_idx
    for file_path in file_list:
        # 이미지 파일 복사
        new_image_name = f"{prefix}{idx:03d}_0000.png"
        new_image_path = os.path.join(image_dest, new_image_name)
        shutil.copy(file_path, new_image_path)
        
        # GT 파일 경로 생성 및 복사
        label_path = file_path.replace('imagesTr', 'normalized')
        new_label_name = f"{prefix}{idx:03d}.png"
        new_label_path = os.path.join(label_dest, new_label_name)
        shutil.copy(label_path, new_label_path)
        
        idx += 1
    return idx

# JSON 파일 경로
json_file_path = './final.json'

# 읽기 JSON 파일
with open(json_file_path, 'r') as f:
    data = json.load(f)

# 폴더 경로 설정
folders = {
    "train": {"images": "./imagesTr", "labels": "./labelsTr"},
    "valid": {"images": "./imagesVal", "labels": "./labelsVal"},
    "test": {"images": "./imagesTest", "labels": "./labelsTest"}
}

# 각 경로에 대해 디렉토리 생성
for folder in folders.values():
    os.makedirs(folder["images"], exist_ok=True)
    os.makedirs(folder["labels"], exist_ok=True)

# Train 파일 복사 및 이름 변경
current_idx = copy_files(data['train'], folders['train']['images'], folders['train']['labels'], prefix='')

# Valid 파일 복사 및 이름 변경 후 Train에 합치기
current_idx = copy_files(data['valid'], folders['train']['images'], folders['train']['labels'], prefix='', start_idx=current_idx)

# Test 파일 복사 및 이름 변경
copy_files(data['test'], folders['test']['images'], folders['test']['labels'], prefix='')

# Valid 디렉토리 비우기
shutil.rmtree(folders['valid']['images'])
shutil.rmtree(folders['valid']['labels'])
os.makedirs(folders['valid']['images'], exist_ok=True)
os.makedirs(folders['valid']['labels'], exist_ok=True)

print("파일 복사 및 합치기가 완료되었습니다.")
