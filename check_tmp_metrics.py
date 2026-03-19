import os
import glob
try:
    from tensorboard.backend.event_processing import event_accumulator
    
    log_dir = "runs/cnn_experiment"
    if not os.path.exists(log_dir):
        print("[-] Thư mục logs TensorBoard chưa được tạo.")
    else:
        log_files = glob.glob(os.path.join(log_dir, "events.out.tfevents*"))
        if not log_files:
            print("[-] Chưa có file log nào được ghi.")
        else:
            latest_log = max(log_files, key=os.path.getctime)
            print(f"[+] Đang đọc file Tạm: {os.path.basename(latest_log)}")
            
            ea = event_accumulator.EventAccumulator(latest_log)
            ea.Reload()
            tags = ea.Tags().get('scalars', [])
            
            if not tags:
                print("[-] Mô hình đang chạy Epoch đầu tiên, chưa có thông số Validations (Validation Metrics) được lưu ra.")
            else:
                print("\n=== KẾT QUẢ TẠM THỜI (LIVE METRICS LATEST EPOCH) ===")
                for tag in tags:
                    events = ea.Scalars(tag)
                    if events:
                        last_event = events[-1]
                        print(f" -> {tag}: Epoch {last_event.step} | Value: {last_event.value:.4f}")
except Exception as e:
    print(f"Lỗi đọc TensorBoard: {e}")
