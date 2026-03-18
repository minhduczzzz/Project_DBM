# 🤖 Chatbot Setup Guide - Hướng Dẫn Cấu Hình Chatbot

## Tính Năng Chatbot

Sau khi dự đoán được giống chó, bạn có thể:
- 💬 Chat trực tiếp với AI chuyên gia về giống chó đó
- 📚 Hỏi về đặc điểm, tính cách, chăm sóc của giống
- 💾 Tự động lưu lịch sử chat vào SQLite
- 🔄 Xem lại các cuộc trò chuyện trước

## 🔑 Bước 1: Lấy Gemini API Key (Miễn Phí)

1. Truy cập: **https://makersuite.google.com/app/apikey**
2. Click **"Create API Key"**
3. Chọn **"Create API key in new Google Cloud project"** (hoặc existing project)
4. Copy API key được tạo
5. Giữ nó ở chỗ an toàn (không chia sẻ công khai)

> ℹ️ **Lưu ý:** Google cung cấp mức free quota đủ để sử dụng chatbot.
> Kiểm tra giới hạn tại: https://ai.google.dev/pricing

---

## 📝 Bước 2: Cấu Hình API Key

### Cách 1: Dùng File .env (Khuyên Dùng - An Toàn Nhất)
1. Copy file: `cp .env.example .env` (hoặc rename `.env.example` thành `.env`)
2. Mở file `.env` bằng text editor
3. Thay `your_api_key_here` bằng API key của bạn:
   ```
   GEMINI_API_KEY=sk-xxxxxxxxxxxxxx
   ```
4. Lưu file
5. Chạy app: `streamlit run demo.py`
6. App sẽ tự động load API key từ `.env`

**Ưu điểm:**
- ✅ An toàn (file .env được ignore bởi .gitignore)
- ✅ Không cần nhập mỗi lần chạy app
- ✅ Không lo lộ key khi push git

### Cách 2: Nhập vào Sidebar (Nếu không dùng .env)
1. Chạy app: `streamlit run demo.py`
2. Trong **sidebar**, mục "🔑 Gemini API Setup"
3. Paste API key vào ô text input
4. Nếu hợp lệ, sẽ hiển thị ✅

### Cách 3: Dùng Environment Variable
Windows PowerShell:
```powershell
$env:GEMINI_API_KEY = "your_api_key_here"
streamlit run demo.py
```

Windows Command Prompt:
```cmd
set GEMINI_API_KEY=your_api_key_here
streamlit run demo.py
```

---

## 🚀 Bước 3: Sử Dụng Chatbot

1. **Upload ảnh chó** → Dự đoán giống chó
2. **Kéo xuống** → Mục "💬 Ask About This Breed"
3. **Nhập câu hỏi** vào ô chat
4. **AI sẽ trả lời** ngay lập tức

Ví dụ câu hỏi:
- *"Golden Retrievers có dễ huấn luyện không?"*
- *"Giống này cần bao nhiêu bài tập hàng ngày?"*
- *"Có vấn đề sức khỏe nào phổ biến không?"*
- *"Giống này phù hợp với gia đình có trẻ em không?"*

---

## 📚 Lịch Sử Chat

### Xem trong Sidebar
- Mục **"📚 Chat History"** hiển thị tất cả các cuộc trò chuyện
- Click **Expander** để xem chi tiết
- Nút **"Delete"** để xóa cuộc trò chuyện

### SQLite Database
Tất cả tin nhắn được lưu trong file `chat_history.db`:
- **Bảng:** `chat_sessions` (thông tin cuộc trò chuyện)
- **Bảng:** `chat_messages` (chi tiết tin nhắn)

---

## 🛠️ Troubleshooting

### ❌ "API Key not found" Error
**Nguyên nhân:** API key không được thiết lập
**Giải pháp:**
1. Kiểm tra environment variable: `echo $env:GEMINI_API_KEY`
2. Hoặc nhập trực tiếp trong sidebar

### ❌ "Invalid API Key" Error
**Nguyên nhân:** API key sai hoặc hết hạn
**Giải pháp:**
1. Tạo key mới tại https://makersuite.google.com/app/apikey
2. Thử lại với key mới

### ❌ Chatbot không phản hồi
**Nguyên nhân:** Quota hết hoặc kết nối mạng
**Giải pháp:**
1. Kiểm tra kết nối internet
2. Chờ 24h để reset quota
3. Kiểm tra usage tại Google AI Studio

### 📚 Cơ sở dữ liệu không lưu
**Giải pháp:**
- Đảm bảo project folder có quyền ghi
- Xóa file `chat_history.db` cũ và tạo lại

---

## 📌 Lưu Ý Bảo Mật

- ⚠️ **Không chia sẻ API key công khai**
- 🔒 Dùng environment variable thay vì hardcode
- 🗑️ Xóa lịch sử chat nếu cần bảo mật

---

## 💡 Mẹo Sử Dụng

1. **System Prompt Tùy Chỉnh**: Chatbot tự động trở thành chuyên gia về giống chó bạn chọn
2. **Câu Hỏi Chi Tiết**: Hỏi cụ thể để được phản hồi chi tiết hơn
3. **Lịch Sử Context**: Chatbot nhớ toàn bộ cuộc trò chuyện trong session

---

## 📖 Tài Liệu Thêm

- Gemini API Docs: https://ai.google.dev/docs
- Streamlit Docs: https://docs.streamlit.io
- SQLite Docs: https://www.sqlite.org/docs.html
