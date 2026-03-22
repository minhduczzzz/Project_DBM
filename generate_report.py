import torch
import pandas as pd
from tqdm import tqdm
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split

from dataset import DogBreedTrainValDataset
from model_vgg import DogBreedVGG16
from transforms import get_val_transform

def generate_slide_report():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Đang load model và dữ liệu...")

    # 1. Load checkpoint để lấy class_to_idx chính xác
    ckpt_path = "training_models/best_vgg.pth"
    checkpoint = torch.load(ckpt_path, map_location=device)
    class_to_idx = checkpoint["class_to_idx"]
    idx_to_class = {v: k for k, v in class_to_idx.items()}
    num_classes = len(class_to_idx)

    # 2. Tái tạo lại đúng tập Test như lúc Train (cùng random_state=42)
    df = pd.read_csv("labels.csv")
    _, test_df = train_test_split(df, test_size=0.2, random_state=42, stratify=df["breed"])

    val_transform = get_val_transform(input_size=224)
    test_dataset = DogBreedTrainValDataset("train", test_df, val_transform, class_to_idx=class_to_idx)
    test_dataloader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=2)

    # 3. Load Model VGG16
    model = DogBreedVGG16(num_classes=num_classes, pretrained=False).to(device)
    model.load_state_dict(checkpoint["model"])
    model.eval()

    # 4. Chạy dự đoán (Inference)
    all_labels = []
    all_preds = []
    print("Đang chấm điểm trên tập Test...")
    with torch.no_grad():
        for images, labels in tqdm(test_dataloader, colour="cyan"):
            images = images.to(device)
            preds = torch.argmax(model(images), dim=1)
            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())

    # 5. Lấy metrics dưới dạng Dictionary để bóc tách dữ liệu
    target_names = [idx_to_class[i] for i in range(num_classes)]
    report_dict = classification_report(all_labels, all_preds, target_names=target_names, output_dict=True, zero_division=0)

    # 6. IN RA KẾT QUẢ ĐÃ FORMAT CHO SLIDE
    print("\n" + "="*70)
    print(" ✂️ COPY KHUNG DƯỚI ĐÂY ĐỂ DÁN VÀO SLIDE (Dùng font Courier New)")
    print("="*70 + "\n")

    acc = report_dict['accuracy']
    macro_f1 = report_dict['macro avg']['f1-score']

    # Header
    print(f"--- VGG16 ---")
    print(f"Accuracy: {acc:.4f}")
    print(f"Macro F1: {macro_f1:.4f}\n")
    print(f"{'':<25} {'precision':>10} {'recall':>10} {'f1-score':>10} {'support':>10}\n")

    # Lọc ra 3 con cao nhất và 3 con thấp nhất dựa theo f1-score
    class_metrics = []
    for cls in target_names:
        class_metrics.append({
            'name': cls,
            'p': report_dict[cls]['precision'],
            'r': report_dict[cls]['recall'],
            'f1': report_dict[cls]['f1-score'],
            's': int(report_dict[cls]['support'])
        })
    class_metrics.sort(key=lambda x: x['f1'], reverse=True)

    # In 3 con tốt nhất
    for m in class_metrics[:3]:
        print(f"{m['name']:>25} {m['p']:10.2f} {m['r']:10.2f} {m['f1']:10.2f} {m['s']:10d}")

    # Dòng chấm chấm ẩn đi phần giữa
    print(f"\n{'... (ẩn 114 giống chó còn lại) ...':>45}\n")

    # In 3 con kém nhất
    for m in class_metrics[-3:]:
        print(f"{m['name']:>25} {m['p']:10.2f} {m['r']:10.2f} {m['f1']:10.2f} {m['s']:10d}")

    print("\n")

    # In phần Averages y hệt như slide mẫu
    total_support = int(report_dict['macro avg']['support'])
    print(f"{'accuracy':>25} {'':10} {'':10} {acc:10.2f} {total_support:10d}")
    
    mac = report_dict['macro avg']
    print(f"{'macro avg':>25} {mac['precision']:10.2f} {mac['recall']:10.2f} {mac['f1-score']:10.2f} {total_support:10d}")
    
    wei = report_dict['weighted avg']
    print(f"{'weighted avg':>25} {wei['precision']:10.2f} {wei['recall']:10.2f} {wei['f1-score']:10.2f} {total_support:10d}")

    print("\n" + "="*70)

if __name__ == "__main__":
    generate_slide_report()