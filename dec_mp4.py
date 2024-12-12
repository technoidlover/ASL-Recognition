import tkinter as tk
from tkinter import filedialog, messagebox
import cv2
import os
import numpy as np
from keras.models import load_model
import time

# Biến toàn cục
video_path = None
roi = None
results = []

# Tải model và cấu hình
model = load_model('models/mymodel.h5')
gesture_names = {0: 'E', 1: 'L', 2: 'F', 3: 'V', 4: 'B'}

# Các tham số
threshold = 60
blurValue = 41
predThreshold = 95

def browse_file():
    global video_path
    video_path = filedialog.askopenfilename(filetypes=[("MP4 files", "*.mp4"), ("All files", "*.*")])
    if video_path:
        messagebox.showinfo("File Selected", f"Selected file: {video_path}")
        set_region()

def set_region():
    global roi
    if not video_path:
        messagebox.showerror("Error", "Hãy chọn file video trước!")
        return

    cap = cv2.VideoCapture(video_path)
    ret, frame = cap.read()
    cap.release()

    if not ret:
        messagebox.showerror("Error", "Không thể đọc file video!")
        return

    roi = cv2.selectROI("Chọn vùng ROI", frame)
    cv2.destroyWindow("Chọn vùng ROI")

    if roi == (0, 0, 0, 0):
        roi = None
        messagebox.showwarning("Warning", "ROI không hợp lệ!")
    else:
        messagebox.showinfo("ROI Set", f"Đã chọn vùng ROI: {roi}")

def predict_sign(image):
    image = np.array(image, dtype='float32')
    image /= 255
    pred_array = model.predict(image)
    result = gesture_names[np.argmax(pred_array)]
    score = float("%0.2f" % (max(pred_array[0]) * 100))
    return result, score

def run_detection():
    if not video_path or not roi:
        messagebox.showerror("Error", "Hãy chọn video và ROI trước!")
        return

    cap = cv2.VideoCapture(video_path)
    results.clear()
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Lấy vùng ROI
        x, y, w, h = roi
        hand_roi = frame[y:y + h, x:x + w]
        
        # Tiền xử lý
        hand_roi = cv2.bilateralFilter(hand_roi, 5, 50, 100)
        gray = cv2.cvtColor(hand_roi, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (blurValue, blurValue), 0)
        ret, thresh = cv2.threshold(blur, threshold, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        # Chuẩn bị cho model
        if np.count_nonzero(thresh)/(thresh.shape[0]*thresh.shape[0]) > 0.2:
            target = np.stack((thresh,) * 3, axis=-1)
            target = cv2.resize(target, (224, 224))
            target = target.reshape(1, 224, 224, 3)
            
            # Dự đoán
            result, score = predict_sign(target)
            
            if score >= predThreshold:
                cv2.putText(frame, f"Ký hiệu: {result}", (50, 50), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                timestamp = cap.get(cv2.CAP_PROP_POS_MSEC)/1000
                results.append(f"Thời điểm: {timestamp:.2f}s - Ký hiệu: {result}")

        cv2.imshow('Kết quả nhận diện', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    messagebox.showinfo("Hoàn thành", "Đã nhận diện xong video!")

def save_results():
    if not results:
        messagebox.showerror("Error", "Chưa có kết quả để lưu!")
        return

    save_path = filedialog.asksaveasfilename(defaultextension=".txt", 
                                            filetypes=[("Text files", "*.txt")])
    if save_path:
        with open(save_path, 'w', encoding='utf-8') as f:
            f.write("\n".join(results))
        messagebox.showinfo("Đã lưu", f"Kết quả đã được lưu vào: {save_path}")

def build_gui():
    root = tk.Tk()
    root.title("Ứng dụng nhận diện thủ ngữ từ video")
    root.geometry("400x200")

    tk.Button(root, text="Chọn Video", command=browse_file).pack(pady=10)
    tk.Button(root, text="Bắt đầu nhận diện", command=run_detection).pack(pady=10)
    tk.Button(root, text="Lưu kết quả", command=save_results).pack(pady=10)
    tk.Button(root, text="Thoát", command=root.quit).pack(pady=10)

    root.mainloop()

if __name__ == "__main__":
    build_gui()