# Workflow: Local + Colab

## Chiến lược

```
Máy Local (CPU)          →    Colab (GPU)
├─ Data Mining                ├─ Train VGG
├─ EDA & Analysis             ├─ Train ResNet
├─ Data Cleaning              ├─ Train EfficientNet
└─ Push to Git                └─ Compare Models
```

## Bước 1: Trên Máy Local (Windows)

### 1.1. Data Mining & Analysis
```bash
# Chạy phân tích dataset
python data_mining_analysis.py
```

**Output**:
- `data_mining_results/mining_report.json` - Insights & recommendations
- `data_mining_results/*.png` - Visualizations
- Hiểu rõ dataset, biết cần augmentation gì

### 1.2. Review Results
```bash
# Xem kết quả
cat data_mining_results/mining_report.json

# Hoặc mở file PNG để xem charts
```

**Quyết định dựa trên insights**:
- Class imbalance → Cần class weights?
- Image sizes → Input size nào phù hợp?
- Dataset size → Augmentation mạnh hay nhẹ?

### 1.3. (Optional) Test code locally
```bash
# Test xem code có lỗi không (chạy 1-2 epochs)
# KHÔNG cần train hết, chỉ test
python train_vgg.py  # Ctrl+C sau 1-2 epochs
```

### 1.4. Commit & Push to Git
```bash
git add .
git commit -m "Add VGG training with data mining results"
git push origin vinhkhoa
```

---

## Bước 2: Trên Colab (GPU)

### 2.1. Clone repo (đã có sẵn trong notebook)
```python
!git clone https://github.com/OWNER/REPO.git
%cd REPO
!git checkout vinhkhoa
```

### 2.2. Kiểm tra GPU
```python
import torch
print(torch.cuda.is_available())  # Phải là True
```

### 2.3. Cài dependencies
```python
!pip install -q -r requirements.txt
```

### 2.4. Upload/Mount dataset
```python
# Option 1: Mount Drive (nếu dataset đã có trên Drive)
from google.colab import drive
drive.mount('/content/drive')
!cp -r /content/drive/MyDrive/dog-breed-dataset/train ./train
!cp /content/drive/MyDrive/dog-breed-dataset/labels.csv ./labels.csv

# Option 2: Upload trực tiếp (nếu dataset nhỏ)
from google.colab import files
# Upload zip file rồi unzip
```

### 2.5. (Optional) Xem lại Data Mining results
```python
# Data mining results đã có trong repo (đã push từ local)
import json
with open('data_mining_results/mining_report.json', 'r') as f:
    report = json.load(f)
print(report)
```

### 2.6. 🚀 TRAIN MODEL (Bước chính)
```python
# Chỉ cần chạy lệnh này!
!python train_vgg.py
```

**Colab sẽ train với GPU, nhanh hơn nhiều!**

### 2.7. View results
```python
# Xem metrics
import json
with open('training_models/vgg_test_metrics.json', 'r') as f:
    metrics = json.load(f)
print(f"Accuracy: {metrics['test_accuracy']:.4f}")
```

### 2.8. Download model về
```python
from google.colab import files
files.download('training_models/best_vgg.pth')
files.download('training_models/vgg_test_metrics.json')
```

### 2.9. Backup to Drive
```python
!cp -r training_models /content/drive/MyDrive/dog_breed_models/
!cp -r tensorboard_vgg /content/drive/MyDrive/dog_breed_tensorboard/
```

---

## Tóm tắt

### Máy Local (1 lần):
1. ✅ Data Mining → Hiểu dataset
2. ✅ Review insights → Quyết định strategy
3. ✅ Push code to Git

### Colab (Mỗi model):
1. ✅ Clone repo
2. ✅ Upload dataset
3. ✅ `!python train_vgg.py` ← CHỈ CẦN LỆNH NÀY!
4. ✅ Download results

---

## Lợi ích

### ✅ Máy Local:
- Phân tích dataset không cần GPU
- Làm việc offline
- Dễ debug và test

### ✅ Colab:
- GPU miễn phí (T4)
- Train nhanh hơn 10-50x
- Không tốn điện máy local

### ✅ Workflow:
- Data Mining chỉ chạy 1 lần (local)
- Training chạy nhiều lần (Colab với GPU)
- Tách biệt rõ ràng: Analysis vs Training

---

## Tips

### 1. Dataset lớn?
- Upload lên Google Drive 1 lần
- Mỗi lần train chỉ cần mount Drive

### 2. Train nhiều models?
```python
# Trên Colab, chạy tuần tự
!python train_vgg.py
!python train_resnet.py
!python train_efficientnet.py

# Rồi compare
!python compare_models.py
```

### 3. Colab timeout?
- Colab free timeout sau 12h
- Lưu checkpoint thường xuyên
- Backup vào Drive định kỳ

### 4. Muốn train tiếp?
```python
# Code đã có checkpoint loading
# Chỉ cần chạy lại, nó sẽ resume từ last epoch
!python train_vgg.py  # Tự động load last_vgg.pth nếu có
```
