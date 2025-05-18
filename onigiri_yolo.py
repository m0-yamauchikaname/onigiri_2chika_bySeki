import cv2
import numpy as np
import time
import os
import csv  # 先頭付近に追加
from ultralytics import YOLO
script_dir = os.path.dirname(os.path.abspath(__file__))
yolo_model_path = os.path.join(script_dir, "yolo_model", "0517.pt")
output_csv_path = os.path.join(script_dir, "Table","output.csv")
# ../yolo_model/0518.pt
if not os.path.exists(yolo_model_path):
    print("YOLOv5モデルが見つかりません。")
    exit(-1)
# YOLOv5モデルのロード
model = YOLO(yolo_model_path)

# タイマー開始
start = time.time()

# --- 0. 入力画像読み込み ---
#input_folder_name= "input_images1\sample4.jpg"
input_folder_name= "input_images2\sample12.jpg"
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
mm_per_pixel = 240 / image_height_pixels  # 縦の長さが2400mmの場合
# 画像の中心座標
center_x = image_width_pixels // 2
center_y = image_height_pixels // 2
print(f"画像の中心座標: ({center_x}, {center_y})")
# 画像で推論
results = model(src)

boxes = results[0].boxes
result_img = src.copy()
found = False

if boxes is not None and len(boxes) > 0:
    # バウンディングボックスの中心座標とインデックスをリスト化
    centers = []
    for i in range(len(boxes)):
        # centersにlabelを追加
        label = int(boxes.cls[i].cpu().numpy())
        conf = float(boxes.conf[i].cpu().numpy())
        xyxy = boxes.xyxy[i].cpu().numpy().astype(int)
        x1, y1, x2, y2 = xyxy
        cx = (x1 + x2) // 2
        cy = (y1 + y2) // 2
        centers.append((label, cx, cy, x1, x2, y1, y2, conf))
    # yの降順でソート
    centers.sort(key=lambda x: x[2])

    # CSVを開く
    with open(output_csv_path, "w", newline="", encoding="utf-8") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["NUMBER", "CRT_POS_X", "CRT_POS_Y"])

        for idx, center in enumerate(centers):
            label, ABScenter_x, ABScenter_y, x1, x2, y1, y2, conf = center
            if label == 0 or label == 1:
                found = True
                # カメラからの相対座標を計算
                RELcenter_x = ABScenter_x - center_x
                RELcenter_y = -(ABScenter_y - center_y)
                # 相対座標をmm単位に変換
                center_x_mm = RELcenter_x * mm_per_pixel
                center_y_mm = RELcenter_y * mm_per_pixel
                # バウンディングボックスの面積を計算
                area = (x2 - x1) * (y2 - y1)
                # バウンディングボックス描画
                cv2.rectangle(result_img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                # バウンディングボックスの中心座標とcountを描画
                cv2.putText(result_img, f"x: {center_x_mm:.2f} y:{center_y_mm:.2f} ", (ABScenter_x, ABScenter_y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
                cv2.putText(result_img, f"count: {idx}", (ABScenter_x, ABScenter_y - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
                cv2.putText(result_img, f"*", (ABScenter_x, ABScenter_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
                # CSVに書き込み
                writer.writerow([idx, f"{center_x_mm:.2f}", f"{center_y_mm:.2f}"])
                # コマンドラインに出力
                print(f"ラベル: {idx}, 信頼度: {conf:.2f}, 座標: ({center_x_mm:.2f}, {center_y_mm:.2f})")
    if found:
        cv2.imwrite(os.path.join(script_dir, "result_label1.jpg"), result_img)
    else:
        print("ラベル1の物体は検出されませんでした。")
else:
    print("物体が検出されませんでした。")