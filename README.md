# Project_DBM - Dog Breed Classification

## Pipeline Overview

```
Dataset
  ↓
📊 DATA MINING & EDA (Optional but Recommended - Phục vụ CẢ DỰ ÁN)
  - Phân tích dataset để đưa ra insights
  - Giúp quyết định augmentation, model selection
  - Không bắt buộc nhưng nên làm trước khi train
  ↓
⚙️ Preprocessing (BẮT BUỘC)
  - Resize images to fixed size
  - Normalize với ImageNet stats
  - Convert to tensor
  ↓
🔄 Augmentation (Recommended)
  - RandomFlip, RandomRotation, ColorJitter
  - Giúp tăng data và giảm overfitting
  ↓
Train / Val / Test Split
  ↓
Feature Extraction (AlexNet | VGG | ResNet | EfficientNet)
  ↓
Training (Fine-tune)
  ↓
Evaluation (Acc, F1, Precision, Recall)
  ↓
Model Comparison
  ↓
Best Model Selection
```

## Vai trò của từng bước

### 1. Data Mining (Optional - Phục vụ CẢ DỰ ÁN)
- **Mục đích**: Hiểu dataset, phát hiện vấn đề, đưa ra recommendations
- **Kết quả**: Insights để chọn augmentation, model, hyperparameters
- **Train được không có?**: ✅ Có, nhưng không biết dataset có vấn đề gì
- **Chạy 1 lần**: Kết quả dùng cho TẤT CẢ models

### 2. Preprocessing (BẮT BUỘC)
- **Mục đích**: Chuẩn hóa input cho model
- **Bao gồm**: Resize, Normalize, ToTensor
- **Train được không có?**: ❌ KHÔNG - Model không nhận input

### 3. Augmentation (Recommended)
- **Mục đích**: Tăng data, giảm overfitting
- **Bao gồm**: RandomFlip, RandomRotation, ColorJitter
- **Train được không có?**: ✅ Có, nhưng accuracy thấp hơn, dễ overfit

## Project Structure

```
Project_DBM/
├── data_mining_analysis.py # Data Mining & EDA
├── dataset.py              # PyTorch Dataset classes (load ảnh, apply transforms)
├── transforms.py           # Transform factory (preprocessing + augmentation)
├── model_vgg.py            # VGG16 model
├── train_vgg.py            # Train VGG16 Phase 1 (freeze features)
├── train_vgg_phase2.py     # Train VGG16 Phase 2 (unfreeze block 5)
├── generate_report.py      # Generate classification report for slides
├── labels.csv              # Dataset labels
├── train/                  # Training images
├── test/                   # Test images
├── training_models/        # Saved models, metrics, and training logs
│   ├── best_vgg.pth
│   ├── vgg_test_metrics.json
│   ├── train_vgg_log_*.txt          # Training logs (auto-generated)
│   ├── train_vgg_phase2_log_*.txt   # Phase 2 logs (auto-generated)
│   └── classification_report_*.txt  # Classification reports (auto-generated)
├── data_mining_results/    # EDA results and visualizations
│   ├── mining_report.json
│   ├── data_mining_log_*.txt        # Data mining logs (auto-generated)
│   └── *.png                        # Visualization charts
└── tensorboard_vgg/        # TensorBoard logs
```

## Kiến trúc Preprocessing

### dataset.py (Dataset Class)
- **Vai trò**: Load ảnh, áp dụng transforms được truyền vào
- **KHÔNG hardcode** preprocessing trong này
- **Lý do**: Mỗi model cần input size khác nhau (VGG: 224, EfficientNet: 240)

### transforms.py (Transform Factory)
- **Vai trò**: Tạo transforms cho các models khác nhau
- **Tái sử dụng**: Không cần viết lại transforms cho mỗi model
- **Flexible**: Dễ dàng thay đổi augmentation strategy

### train_*.py (Training Scripts)
- **Vai trò**: Chọn transform phù hợp, truyền vào Dataset
- **VGG**: `get_train_transform(input_size=224)`
- **EfficientNet**: `get_train_transform(input_size=240)`

### Ví dụ:
```python
# transforms.py tạo transform
train_transform = get_train_transform(input_size=224, augmentation=True)

# dataset.py nhận transform và áp dụng
dataset = DogBreedTrainValDataset(..., transform=train_transform)

# train_vgg.py sử dụng dataset
dataloader = DataLoader(dataset, batch_size=16)
```

## Quick Start

### Step 1: Data Mining & Analysis (Optional but Recommended)

```bash
python data_mining_analysis.py
```

Output:
- Console output + saved to `data_mining_results/data_mining_log_YYYYMMDD_HHMMSS.txt`
- Analysis report: `data_mining_results/mining_report.json`
- Visualizations: `data_mining_results/*.png`
- Distribution data: `data_mining_results/breed_distribution.csv`

This step helps you understand the dataset and make informed decisions about augmentation and model selection.

### Step 2: Training VGG16 (2 Phases)

#### Phase 1: Train Classifier Only (Freeze CNN Features)
```bash
python train_vgg.py
```

Output:
- Console output + saved to `training_models/train_vgg_log_YYYYMMDD_HHMMSS.txt`
- Model: `training_models/best_vgg.pth`
- Metrics: `training_models/vgg_test_metrics.json`
- TensorBoard: `tensorboard_vgg/`

#### Phase 2: Fine-tune Block 5 (Optional - for better accuracy)
```bash
python train_vgg_phase2.py
```

Output:
- Console output + saved to `training_models/train_vgg_phase2_log_YYYYMMDD_HHMMSS.txt`
- Model: `training_models/best_vgg_phase2.pth`

#### Generate Classification Report for Slides
```bash
python generate_report.py
```

Output:
- Console output + saved to `training_models/classification_report_YYYYMMDD_HHMMSS.txt`
- Formatted report ready to copy-paste into presentation slides

## Key Features

### Automatic Logging
All training scripts automatically save their console output to timestamped log files in `training_models/`:
- `train_vgg_log_YYYYMMDD_HHMMSS.txt` - Phase 1 training log
- `train_vgg_phase2_log_YYYYMMDD_HHMMSS.txt` - Phase 2 training log
- `classification_report_YYYYMMDD_HHMMSS.txt` - Classification report

This makes it easy to review training history and share results.

## Monitoring with TensorBoard

```bash
tensorboard --logdir tensorboard_vgg
```

## Requirements

```
torch
torchvision
pandas
scikit-learn
tqdm
tensorboard
pillow
matplotlib
seaborn
```
