# Project_DBM - Dog Breed Classification - VGG model

## Pipeline Overview

```
Dataset
  ↓
📊 DATA MINING & EDA 
  - Phân tích dataset để đưa ra insights
  - Giúp quyết định augmentation, model selection
  - Không bắt buộc nhưng nên làm trước khi train
  ↓
⚙️ Preprocessing
  - Resize images to fixed size
  - Normalize với ImageNet stats
  - Convert to tensor
  ↓
🔄 Augmentation
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

### 1. Data Mining
- **Mục đích**: Hiểu dataset, phát hiện vấn đề, đưa ra recommendations
- **Kết quả**: Insights để chọn augmentation, model, hyperparameters
- **Train được không có?**: ✅ Có, nhưng không biết dataset có vấn đề gì
- **Chạy 1 lần**: Kết quả dùng cho TẤT CẢ models

### 2. Preprocessing 
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
├── train_vgg.py            # Train VGG16 (with detailed pipeline output)
├── compare_models.py       # Compare all models
├── labels.csv              # Dataset labels
├── train/                  # Training images
├── training_models/        # Saved models and metrics
└── data_mining_results/    # EDA results and visualizations
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

### Workflow: Local + Colab (Recommended)

#### Trên Máy Local (1 lần):
```bash
# 1. Data Mining & Analysis
python data_mining_analysis.py

# 2. Review insights
cat data_mining_results/mining_report.json

# 3. Push to Git
git add .
git commit -m "Add data mining results"
git push origin vinhkhoa
```

#### Trên Colab (Train với GPU):
1. Upload `colab_train_only.ipynb` lên Colab
2. Bật GPU: Runtime > Change runtime type > GPU (T4)
3. Chạy từng cell:
   - Clone repo
   - Upload dataset
   - `!python train_vgg.py` ← CHỈ CẦN LỆNH NÀY!
   - Download results

**Xem chi tiết**: [LOCAL_WORKFLOW.md](LOCAL_WORKFLOW.md)

---

### Alternative: All on Local (Nếu có GPU)

```bash
# 1. Data Mining
python data_mining_analysis.py

# 2. Train
python train_vgg.py

# 3. Compare
python compare_models.py
```

## Data Mining & EDA (Phục vụ CẢ DỰ ÁN)

Run exploratory data analysis first:

```bash
python data_mining_analysis.py
```

### Output:
- **Dataset Statistics**: Total samples, number of classes, distribution
- **Class Imbalance Analysis**: Imbalance ratio, underrepresented classes
- **Image Properties**: Dimensions, aspect ratios, file sizes
- **Data Quality**: Missing/corrupted images
- **Visualizations**: Distribution charts, scatter plots
- **Insights & Recommendations**: Actionable insights for ALL models

### Generated Files:
- `data_mining_results/mining_report.json` - Complete analysis report
- `data_mining_results/breed_distribution.csv` - Class distribution
- `data_mining_results/class_distribution.png` - Class distribution charts
- `data_mining_results/image_properties.png` - Image property analysis
- `data_mining_results/dimensions_scatter.png` - Dimension scatter plot

**Quan trọng**: Kết quả Data Mining dùng chung cho TẤT CẢ models (VGG, ResNet, AlexNet, EfficientNet). Chỉ cần chạy 1 lần!

## Training VGG16

```bash
python train_vgg.py
```

### Output Structure:
- **STEP 1**: Dataset loading statistics
- **STEP 2**: Preprocessing and augmentation details
- **STEP 3**: Train/Val/Test split information
- **STEP 4**: Model architecture and parameters
- **STEP 5**: Training progress with loss and accuracy
- **STEP 6**: Final evaluation metrics (Accuracy, Precision, Recall, F1)

### Generated Files:
- `training_models/best_vgg.pth` - Best model checkpoint
- `training_models/vgg_test_metrics.json` - Test metrics
- `training_models/vgg_training_history.json` - Training history
- `tensorboard_vgg/` - TensorBoard logs

## Model Comparison

After training multiple models, compare them:

```bash
python compare_models.py
```

### Output:
- Performance comparison table
- Best model identification
- `training_models/model_comparison.csv` - Comparison table
- `training_models/model_comparison.png` - Visualization charts
- `training_models/metrics_comparison.png` - Metrics comparison

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
