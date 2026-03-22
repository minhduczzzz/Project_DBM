# Data Mining & Exploratory Data Analysis Guide

## Tại sao cần Data Mining?

Data Mining là bước quan trọng trong pipeline Machine Learning để:
1. **Hiểu dữ liệu**: Nắm rõ đặc điểm, phân bố của dataset
2. **Phát hiện vấn đề**: Class imbalance, missing data, corrupted files
3. **Đưa ra quyết định**: Chọn model, augmentation, hyperparameters phù hợp
4. **Tối ưu hiệu suất**: Dựa trên insights để cải thiện model

## Các bước Data Mining trong project

### 1. Dataset Statistics
- Tổng số samples
- Số lượng classes
- Phân bố samples per class (min, max, mean, median, std)
- Missing values
- Duplicate data

**Mục đích**: Hiểu tổng quan về dataset

### 2. Class Imbalance Analysis
- Tính imbalance ratio (max_samples / min_samples)
- Phân loại classes: underrepresented, well-represented, overrepresented
- Xác định classes cần augmentation mạnh hơn

**Mục đích**: Phát hiện vấn đề mất cân bằng dữ liệu

**Ví dụ**:
- Imbalance ratio > 2: Cần class weights hoặc oversampling
- Nhiều classes underrepresented: Cần augmentation mạnh

### 3. Image Properties Analysis
- Kích thước ảnh (width, height)
- Aspect ratio (tỷ lệ khung hình)
- File size
- Phân bố các thuộc tính

**Mục đích**: Quyết định input size, augmentation strategies

**Ví dụ**:
- Ảnh có nhiều aspect ratio khác nhau → Cần resize cẩn thận
- File size lớn → Có thể cần compression

### 4. Data Quality Check
- Missing images (file không tồn tại)
- Corrupted images (file bị lỗi)
- Valid images

**Mục đích**: Đảm bảo chất lượng dữ liệu trước khi train

### 5. Visualizations
- Class distribution charts (top/bottom breeds, histogram, boxplot)
- Image properties charts (width, height, aspect ratio, file size)
- Scatter plots (dimensions correlation)

**Mục đích**: Trực quan hóa để dễ hiểu và phát hiện patterns

### 6. Insights & Recommendations
Dựa trên phân tích, đưa ra:
- **Insights**: Những phát hiện quan trọng
- **Recommendations**: Đề xuất cụ thể cho training

**Ví dụ Recommendations**:
- "Use class weights for imbalanced classes"
- "Apply transfer learning with pretrained models"
- "Use label smoothing to prevent overfitting"

## Kết quả Data Mining

### Files được tạo:
```
data_mining_results/
├── mining_report.json          # Báo cáo tổng hợp
├── breed_distribution.csv      # Phân bố classes
├── class_distribution.png      # Charts phân bố classes
├── image_properties.png        # Charts thuộc tính ảnh
└── dimensions_scatter.png      # Scatter plot kích thước
```

### Cách sử dụng kết quả:

1. **Đọc mining_report.json** để hiểu tổng quan
2. **Xem visualizations** để phát hiện patterns
3. **Áp dụng recommendations** khi train model:
   - Chọn augmentation phù hợp
   - Set class weights nếu cần
   - Chọn model architecture phù hợp
   - Điều chỉnh hyperparameters

## Ví dụ áp dụng

### Scenario 1: High Class Imbalance
**Phát hiện**: Imbalance ratio = 5.2
**Recommendation**: 
- Sử dụng class weights trong loss function
- Áp dụng oversampling cho minority classes
- Augmentation mạnh hơn cho underrepresented classes

### Scenario 2: Small Dataset
**Phát hiện**: Chỉ có 5000 samples, 120 classes
**Recommendation**:
- Sử dụng transfer learning (VGG, ResNet, EfficientNet)
- Pretrained weights từ ImageNet
- Extensive data augmentation
- Early stopping để tránh overfitting

### Scenario 3: Varied Image Sizes
**Phát hiện**: Width từ 200-4000px, aspect ratio từ 0.5-2.5
**Recommendation**:
- Resize về 224x224 (standard cho pretrained models)
- Sử dụng padding để giữ aspect ratio nếu cần
- Augmentation: RandomCrop, RandomResizedCrop

## Tích hợp vào Pipeline

```python
# 1. Data Mining (TRƯỚC KHI TRAIN)
python data_mining_analysis.py

# 2. Đọc insights và recommendations
# 3. Điều chỉnh training config dựa trên insights

# 4. Train model
python train_vgg.py

# 5. Evaluate và compare
python compare_models.py
```

## Metrics quan trọng

### Dataset Level:
- **Imbalance Ratio**: Đo mức độ mất cân bằng
- **Samples per Class**: Phân bố dữ liệu
- **Data Quality Score**: % valid images

### Image Level:
- **Mean Dimensions**: Kích thước trung bình
- **Aspect Ratio Distribution**: Phân bố tỷ lệ khung hình
- **File Size Statistics**: Dung lượng file

## Best Practices

1. **Luôn chạy Data Mining trước khi train**
2. **Lưu lại mining report** để tham khảo sau này
3. **So sánh mining results** giữa các datasets
4. **Update recommendations** khi có dataset mới
5. **Document insights** trong báo cáo cuối cùng

## Kết luận

Data Mining không chỉ là bước phân tích, mà là nền tảng để:
- Hiểu rõ dữ liệu
- Đưa ra quyết định đúng đắn
- Tối ưu hóa model performance
- Giải thích kết quả training

**→ Đây chính là cốt lõi của Data Mining trong Machine Learning project!**
