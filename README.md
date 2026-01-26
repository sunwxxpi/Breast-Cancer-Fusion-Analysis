# Breast Cancer Fusion Analysis

유방암 검사 데이터의 멀티모달 융합 분석을 위한 딥러닝 프로젝트입니다. 초음파 영상의 B-모드(B-Mode)와 탄성 변형률 영상(Strain Elastography/SE-Mode) 두 가지 모달리티를 결합하여 종양 분류를 수행합니다.

## 주요 특징

- **멀티모달 융합**: B-모드(구조적 정보)와 SE-모드(탄성 정보)를 결합한 조기 융합(Early Fusion) 방식
- **DenseNet-161 백본**: 사전학습 가중치를 활용한 전이 학습
- **K-Fold 교차검증**: 데이터 부족 상황에서 신뢰성 있는 성능 평가
- **3-클래스 분류**: 정상(Normal), 양성(Benign), 악성(Malignant)

## 프로젝트 구조

```
Breast-Cancer-Fusion-Analysis/
├── model.py                 # 멀티모달 통합 모델 정의
├── dataset.py               # 데이터셋 로더 클래스
├── train.py                 # K-Fold 교차검증 학습 스크립트
├── test.py                  # 모델 평가 및 혼동 행렬 시각화
│
├── preprocessing/           # 전처리 Jupyter 노트북
│   ├── BUSI.ipynb          # BUSI 데이터셋 전처리
│   ├── STU.ipynb           # STU 데이터셋 전처리
│   └── TDSC-ABUS2023.ipynb # TDSC-ABUS2023 데이터셋 전처리
│
└── util/                    # 유틸리티 스크립트
    ├── square_padding.py    # 이미지 정사각형 패딩
    ├── seg_preprocess/      # 세그멘테이션 전처리
    └── strain_preprocess/   # 탄성 영상 전처리
```

## 데이터셋

### 지원 데이터셋
- **BUSI** (Breast Ultrasound Images)
- **STU** (Strain Ultrasound)
- **TDSC-ABUS2023** (3D Automated Breast Ultrasound)

### 데이터 준비
```bash
ln -s /path/to/your_dataset dataset
```

### 데이터 구조
```
data/
├── train/
│   ├── busi_elastogram_gray/    # B-모드 (그레이스케일)
│   │   ├── benign/
│   │   ├── malignant/
│   │   └── normal/
│   └── busi_elastogram/         # SE-모드 (컬러)
│       ├── benign/
│       ├── malignant/
│       └── normal/
└── test/
    ├── busi_elastogram_gray/
    └── busi_elastogram/
```

## 모델 아키텍처

### Stack-wise Integration Model

```
B-Mode Input (3ch)     SE-Mode Input (3ch)
        ↓                       ↓
    [Concatenate]  → 6-channel 입력
        ↓
   [Conv1×1]  → 3-channel 통합
        ↓
  [DenseNet-161 Backbone]
        ↓
  [Classifier FC]
        ↓
  Output (3-class)
```

### 모달리티 선택 옵션
- `mode='b_mode'`: B-모드만 사용
- `mode='se_mode'`: SE-모드만 사용
- `mode='both'`: 두 모달리티 통합

## 요구사항

```
torch
torchvision
scikit-learn
opencv-python
Pillow
pydicom
numpy
matplotlib
seaborn
tqdm
ttach
```

## 사용법

### 학습
```bash
python train.py
```

**주요 하이퍼파라미터** (train.py에서 수정 가능):
| 파라미터 | 값 | 설명 |
|----------|-----|------|
| `n_splits` | 5 | K-Fold 분할 수 |
| `batch_size` | 32 | 배치 크기 |
| `num_epochs` | 300 | 학습 에포크 |
| `learning_rate` | 0.0002 | 초기 학습률 |
| `warmup_epochs` | 10 | Warm-up 기간 |
| `mode` | 'b_mode' | 모달리티 선택 |

### 평가
```bash
python test.py
```
저장된 모델 체크포인트(`.pt`)를 자동으로 감지하여 테스트를 수행합니다.

## 학습 파이프라인

1. **데이터 증강**
   - 448×448 리사이징 (Bicubic 보간)
   - 무작위 수평 뒤집기 (p=0.5)
   - 정규화: mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]

2. **옵티마이저**: AdamW
3. **손실함수**: CrossEntropyLoss (Label Smoothing=0.15)
4. **학습률 스케줄러**: Warm-up + Cosine Annealing

## 전처리 유틸리티

### 세그멘테이션 전처리 (`util/seg_preprocess/`)
| 파일 | 기능 |
|------|------|
| `binary_mask.py` | 마스크 이미지 이진화 |
| `normalize_to_zero_one.py` | 픽셀값 정규화 (0-1) |
| `resize.py` | 이미지와 레이블 크기 조정 |
| `to_grayscale.py` | 그레이스케일 변환 |

### 탄성 영상 전처리 (`util/strain_preprocess/`)
| 파일 | 기능 |
|------|------|
| `dcm_to_png_converter.py` | DICOM → PNG 변환 |
| `elastogram_extractor.py` | 탄성 영상 ROI 추출 |
| `files_organizer.py` | 파일 분류 (Grade/Score) |

## 결과 출력

- **모델 체크포인트**: `fold_{fold}__{epoch}_{val_loss:.4f}.pt`
- **설정 파일**: `config.txt`
- **혼동 행렬**: PNG 이미지로 저장
- **평가 메트릭**: 정확도, F1-Score, 분류 리포트

## 참고 자료

- [BUSI Dataset](https://www.kaggle.com/datasets/sabahesaraki/breast-ultrasound-images-dataset)
- [TDSC-ABUS2023 Challenge](https://tdsc-abus2023.grand-challenge.org/)
