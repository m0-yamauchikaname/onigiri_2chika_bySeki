import cv2
import numpy as np
import time
import os
from ultralytics import YOLO
script_dir = os.path.dirname(os.path.abspath(__file__))

# ../yolo_model/0516.pt
yolo_model_path = os.path.join(script_dir, "yolo_model", "0516.pt")
if not os.path.exists(yolo_model_path):
    print("YOLOv5モデルが見つかりません。")
    exit(-1)
# YOLOv5モデルのロード
model = YOLO(yolo_model_path)

# タイマー開始
start = time.time()

# --- 0. 入力画像読み込み ---
input_folder_name= "input_images1\sample3.jpg"
#input_folder_name= "input_images2\sample12.jpg"
folder_path = os.path.join(script_dir, input_folder_name)
src_full = cv2.imread(folder_path, cv2.IMREAD_COLOR)
if src_full is None:
    print("画像が見つかりません。")
    exit(-1)
# 中心からトリミング
center_x = src_full.shape[1] // 2
center_y = src_full.shape[0] // 2
#crop_width = 3840
crop_width = 1200
crop_height = 2160

x = max(center_x - crop_width // 2, 0)
y = max(center_y - crop_height // 2, 0)
x2 = min(x + crop_width, src_full.shape[1])
y2 = min(y + crop_height, src_full.shape[0])

src = src_full[y:y2, x:x2].copy()

# 画像の縦方向のピクセル数
image_height_pixels = src.shape[0]
image_width_pixels = src.shape[1]

# 1ピクセルあたりの物理的な長さ（mm）
mm_per_pixel = 2400 / image_height_pixels  # 縦の長さが2400mmの場合
# 画像の中心座標
center_x = image_width_pixels // 2
center_y = image_height_pixels // 2
print(f"画像の中心座標: ({center_x}, {center_y})")
# 画像で推論
results = model(src)

# ラベルが1の物体のみ出力＆バウンディングボックスを画像に描画
boxes = results[0].boxes
result_img = src.copy()
found = False
count = 0
if boxes is not None and len(boxes) > 0:
    for i in range(len(boxes)):
        label = int(boxes.cls[i].cpu().numpy())
        if label == 1:
            count += 1
            found = True
            conf = float(boxes.conf[i].cpu().numpy())
            xyxy = boxes.xyxy[i].cpu().numpy().astype(int)
            x1, y1, x2, y2 = xyxy
            # バウンディングボックスの中心の座標を計算
            Bcenter_x = (x1 + x2) // 2-center_x
            Bcenter_y = -((y1 + y2) // 2-center_y)
            # mm単位に変換
            center_x_mm = Bcenter_x * mm_per_pixel
            center_y_mm = Bcenter_y * mm_per_pixel
            # バウンディングボックスの面積を計算
            area = (x2 - x1) * (y2 - y1)
            # バウンディングボックス描画
            cv2.rectangle(result_img, (x1, y1), (x2, y2), (0, 255, 0), 3)
            cv2.putText(result_img, f"Class:1 Conf:{conf:.2f}", (x1, y1-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
            print(f"ラベル: {count}, 信頼度: {conf:.2f}, 座標: ({center_x_mm:.2f}, {center_y_mm:.2f})")
    if found:
        cv2.imwrite(os.path.join(script_dir, "result_label1.jpg"), result_img)
    else:
        print("ラベル1の物体は検出されませんでした。")
else:
    print("物体が検出されませんでした。")