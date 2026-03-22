# Workflow - Dog Breed Classification

## TL;DR - Quy trình làm việc

### Bước 1: Data Mining (Optional - 1 lần cho CẢ DỰ ÁN)
```bash
python data_mining_analysis.py
```
- ✅ Phân tích dataset
- ✅ Phát hiện vấn đề
- ✅ Nhận recommendations
- 📊 Kết quả dùng cho TẤT CẢ models

### Bước 2: Train Model (VGG16)
```bash
python train_vgg.py
```
- ✅ Preprocessing (BẮT BUỘC): Resize, Normalize
- ✅ Augmentation (Recommended): Flip, Rotation, ColorJitter
- ✅ Training với detailed output

### Bước 3: Compare Models
```bash
python compare_models.py
```
- ✅ So sánh performance
- ✅ Chọn best model

---

## FAQ

### Q1: Data Mining có bắt buộc không?
**A**: Không bắt buộc, nhưng HIGHLY RECOMMENDED. Giúp bạn:
- Hiểu dataset
- Phát hiện vấn đề sớm
- Train hiệu quả hơn

### Q2: Data Mining chỉ cho VGG hay cho cả dự án?
**A**: Cho CẢ DỰ ÁN! Kết quả dùng chung cho VGG, ResNet, AlexNet, EfficientNet. Chỉ cần chạy 1 lần.

### Q3: Không có Preprocessing thì train được không?
**A**: KHÔNG! Preprocessing (Resize, Normalize) là BẮT BUỘC. Model không nhận input nếu thiếu.

### Q4: Không có Augmentation thì train được không?
**A**: Train được, nhưng:
- Accuracy thấp hơn
- Dễ overfit
- Không tận dụng hết data

### Q5: Tại sao cần Normalize theo ImageNet stats?
**A**: Vì dùng pretrained weights từ ImageNet. Model đã học với stats này, phải giữ nguyên.

---

## Chi tiết từng bước

### Data Mining Output
```
data_mining_results/
├── mining_report.json          # Insights & recommendations
├── breed_distribution.csv      # Class distribution
├── class_distribution.png      # Charts
├── image_properties.png        # Image analysis
└── dimensions_scatter.png      # Scatter plot
```

### Training Output
```
training_models/
├── best_vgg.pth               # Best model checkpoint
├── last_vgg.pth               # Last checkpoint
├── vgg_test_metrics.json      # Test metrics
└── vgg_training_history.json  # Training history

tensorboard_vgg/               # TensorBoard logs
```

### Comparison Output
```
training_models/
├── model_comparison.csv       # Comparison table
├── model_comparison.png       # Charts
└── metrics_comparison.png     # Metrics chart
```

---

## Trên Colab

Upload `colab_train_vgg.ipynb` và chạy từng cell:

1. Clone repo + checkout nhánh
2. Bật GPU (Runtime > Change runtime type > GPU)
3. Cài dependencies
4. Upload dataset
5. **Data Mining** (Cell 6)
6. **Train VGG** (Cell 7)
7. View results, plots, TensorBoard
8. Download hoặc backup to Drive

---

## Tóm tắt

| Bước | Bắt buộc? | Chạy bao nhiêu lần? | Mục đích |
|------|-----------|---------------------|----------|
| Data Mining | Optional | 1 lần (cho cả dự án) | Hiểu dataset, insights |
| Preprocessing | BẮT BUỘC | Mỗi lần train | Chuẩn hóa input |
| Augmentation | Recommended | Mỗi lần train | Tăng data, giảm overfit |
| Training | BẮT BUỘC | Mỗi model | Train model |
| Comparison | Optional | 1 lần (sau khi train xong) | Chọn best model |


---

## Kiến trúc Preprocessing

### Q: Dataset.py có làm preprocessing không?
**A**: KHÔNG! Dataset.py chỉ:
- Load ảnh từ disk
- Áp dụng transform được truyền vào
- Return (image, label)

### Q: Vậy preprocessing ở đâu?
**A**: Ở `transforms.py` và được truyền vào Dataset:

```python
# transforms.py - Định nghĩa transforms
train_transform = get_train_transform(input_size=224, augmentation=True)

# dataset.py - Nhận và áp dụng
dataset = DogBreedTrainValDataset(..., transform=train_transform)

# train_vgg.py - Sử dụng
dataloader = DataLoader(dataset, batch_size=16)
```

### Q: Tại sao không hardcode trong dataset.py?
**A**: Vì mỗi model cần input size khác nhau:
- VGG16: 224x224
- EfficientNet-B0: 240x240
- AlexNet: 227x227

Nếu hardcode → Không flexible!

### Q: Mỗi model phải định nghĩa transform riêng?
**A**: KHÔNG! Dùng `transforms.py`:
```python
# VGG
from transforms import VGG_TRAIN_TRANSFORM
dataset = DogBreedTrainValDataset(..., transform=VGG_TRAIN_TRANSFORM)

# EfficientNet
from transforms import EFFICIENTNET_TRAIN_TRANSFORM
dataset = DogBreedTrainValDataset(..., transform=EFFICIENTNET_TRAIN_TRANSFORM)
```
