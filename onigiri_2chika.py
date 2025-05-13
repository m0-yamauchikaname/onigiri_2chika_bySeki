import cv2
import numpy as np
import time
import os

# タイマー開始
start = time.time()

# --- 0. 入力画像読み込み ---
#input_folder_name= "input_images1\sample7.jpg"
input_folder_name= "input_images2\sample12.jpg"
script_dir = os.path.dirname(os.path.abspath(__file__))
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
# コントラスト調整
alpha = 0.9  # コントラスト係数, 0-3
beta = 50 # 明るさ係数, 0-100
#alpha = 1  # 0-3の範囲に制限
#beta = 0  # 0-100の範囲に制限
Contrasted_src = cv2.convertScaleAbs(src, alpha=alpha, beta=beta)

# グレースケール変換
gray = cv2.cvtColor(Contrasted_src, cv2.COLOR_BGR2GRAY)

# --- 1. 二値化（自分で閾値を決める） ---
threshold_value = 180
_, otsu_binary = cv2.threshold(gray, threshold_value, 255, cv2.THRESH_BINARY)

# --- 2. モルフォロジーオープニング（ノイズ除去） ---
kernel_o = cv2.getStructuringElement(cv2.MORPH_RECT, (25, 25))
morph_open = cv2.morphologyEx(otsu_binary, cv2.MORPH_OPEN, kernel_o)

# --- 3. モルフォロジークロージング（穴埋め） ---
kernel_c = cv2.getStructuringElement(cv2.MORPH_RECT, (10, 10))
morph_close = cv2.morphologyEx(morph_open, cv2.MORPH_CLOSE, kernel_c)

# --- 4. HSV変換＋赤色領域抽出 ---
hsv = cv2.cvtColor(src, cv2.COLOR_BGR2HSV)
lower_red_mask = cv2.inRange(hsv, (0, 80, 100), (10, 255, 255))
upper_red_mask = cv2.inRange(hsv, (170, 80, 100), (180, 255, 255))
red_mask = cv2.bitwise_or(lower_red_mask, upper_red_mask)
# --- 5. 赤色領域モルフォロジーオープニング ---
kernel_red = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
red_mask_open = cv2.morphologyEx(red_mask, cv2.MORPH_OPEN, kernel_red)
# --- 6. 赤色領域モルフォロジークロージング ---
kernel_red_c = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 8))
red_mask_close = cv2.morphologyEx(red_mask_open, cv2.MORPH_CLOSE, kernel_red_c)

# --- 7. 赤色領域面積フィルタ ---
# 赤色領域のラベリング
nLabels_red, labels_red = cv2.connectedComponents(red_mask_close)
# 面積フィルタの閾値を設定
red_pixel_threshold_min = 15
red_pixel_threshold_max = 60000
filtered_red = np.zeros_like(labels_red, dtype=np.uint8)
for i in range(1, nLabels_red):
    mask = (labels_red == i).astype(np.uint8)
    overlap = cv2.bitwise_and(red_mask_close, red_mask_close, mask=mask)
    if cv2.countNonZero(overlap) > 0:
        #filtered[labels == i] = 255
        if cv2.countNonZero(mask) > red_pixel_threshold_min^2 and cv2.countNonZero(mask) < red_pixel_threshold_max^2:
            filtered_red[labels_red == i] = 255

# --- 8. ラベリング ---
nLabels, labels = cv2.connectedComponents(morph_close)

# --- 9. 赤色を含むラベルだけ抽出 ---
# ラベルのピクセル数の閾値を設定
pixel_threshold_min = 10000
pixel_threshold_max = 600000
filtered = np.zeros_like(labels, dtype=np.uint8)
for i in range(1, nLabels):
    mask = (labels == i).astype(np.uint8)
    overlap = cv2.bitwise_and(filtered_red, filtered_red, mask=mask)
    if cv2.countNonZero(overlap) > 0:
        #filtered[labels == i] = 255
        if cv2.countNonZero(mask) > pixel_threshold_min^2 and cv2.countNonZero(mask) < pixel_threshold_max^2:
            filtered[labels == i] = 255



# --- ラベリングし直し＆ID表示 ---
nLabels_filtered, labels_filtered = cv2.connectedComponents(filtered)

label_display = cv2.cvtColor(filtered, cv2.COLOR_GRAY2BGR)
src_with_labels = src.copy()

# 画像の縦方向のピクセル数
image_height_pixels = src.shape[0]
image_width_pixels = src.shape[1]

# 1ピクセルあたりの物理的な長さ（mm）
mm_per_pixel = 240 / image_height_pixels  # 縦の長さが2400mmの場合

# 画像の中心座標
center_x = image_width_pixels // 2
center_y = image_height_pixels // 2

# cx, cyの座標を物理単位に変換し、中心を原点とした相対座標を出力
for i in range(1, nLabels_filtered):
    mask = (labels_filtered == i).astype(np.uint8)
    m = cv2.moments(mask)
    if m["m00"] != 0:
        cx = int(m["m10"] / m["m00"])
        cy = int(m["m01"] / m["m00"])
        
        # 中心を原点とした相対座標
        relative_cx = cx - center_x
        relative_cy = -1*(cy - center_y)
        
        # 相対座標を物理単位に変換
        relative_cx_mm = relative_cx * mm_per_pixel
        relative_cy_mm = relative_cy * mm_per_pixel
        
        print(f"Label {i}: relative_cx = {relative_cx} px ({relative_cx_mm:.2f} mm), relative_cy = {relative_cy} px ({relative_cy_mm:.2f} mm)")
        
        # ラベル表示
        cv2.putText(label_display, "*", (cx, cy), cv2.FONT_HERSHEY_SIMPLEX, 3.0, (0, 0, 255), 2)
        cv2.putText(label_display, str(i), (cx + 50, cy), cv2.FONT_HERSHEY_SIMPLEX, 3.0, (0, 0, 255), 10)
        cv2.putText(src_with_labels, "*", (cx, cy), cv2.FONT_HERSHEY_SIMPLEX, 3.0, (0, 255, 0), 2)
        cv2.putText(src_with_labels, str(i), (cx + 50, cy), cv2.FONT_HERSHEY_SIMPLEX, 3.0, (0, 0, 255), 10)
        
        # 座標値を描画
        coord_text = f"({relative_cx_mm:.2f}, {relative_cy_mm:.2f})"
        cv2.putText(src_with_labels, coord_text, (cx + 50, cy + 50), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0), 2)

# 原点に〇を描画
cv2.circle(src_with_labels, (center_x, center_y), 20, (0, 0, 0), 3)  # 青色の円を描画

# --- 表示と保存 ---
def show_resized(win_name, img, idx):
    display_width = 200
    resized_height = int(display_width * img.shape[0] / img.shape[1])
    resized = cv2.resize(img, (display_width, resized_height))
    x = (idx % 3) * (display_width + 10) + 20
    y = (idx // 3) * (display_width + 40) + 50
    cv2.namedWindow(win_name, cv2.WINDOW_AUTOSIZE)
    cv2.moveWindow(win_name, x, y)
    cv2.imshow(win_name, resized)

show_resized("Original (Cropped)", src, 0)
show_resized("Lightened", Contrasted_src, 0)
show_resized("Gray", gray, 1)
show_resized("Otsu Binary", otsu_binary, 2)
show_resized("Morph Open", morph_open, 3)
show_resized("Morph Close", morph_close, 4)
show_resized("Red Mask (HSV)", red_mask, 5)
show_resized("Red Mask Open", red_mask_open, 5)
show_resized("Red Mask Close", red_mask_close, 5)
show_resized("Filtered Red Mask", filtered_red, 5)
show_resized("Filtered Labels", filtered, 6)
show_resized("Labels with ID", label_display, 7)
show_resized("Original with Labels", src_with_labels, 8)

# --- 画像保存 ---
# 保存
folder_path = os.path.join(script_dir, "output_images")
cv2.imwrite(os.path.join(folder_path, "output_original_cropped.png"), src)
cv2.imwrite(os.path.join(folder_path, "output_lightened.png"), Contrasted_src)
cv2.imwrite(os.path.join(folder_path, "output_gray.png"), gray)
cv2.imwrite(os.path.join(folder_path, "output_otsu_binary.png"), otsu_binary)
cv2.imwrite(os.path.join(folder_path, "output_morph_open.png"), morph_open)
cv2.imwrite(os.path.join(folder_path, "output_morph_close.png"), morph_close)
cv2.imwrite(os.path.join(folder_path, "output_red_mask.png"), red_mask)
cv2.imwrite(os.path.join(folder_path, "output_red_mask_open.png"), red_mask_open)
cv2.imwrite(os.path.join(folder_path, "output_red_mask_close.png"), red_mask_close)
cv2.imwrite(os.path.join(folder_path, "output_filtered_red_mask.png"), filtered_red)
cv2.imwrite(os.path.join(folder_path, "output_filtered_labels.png"), filtered)
cv2.imwrite(os.path.join(folder_path, "output_labels_with_id.png"), label_display)
cv2.imwrite(os.path.join(folder_path, "output_original_with_labels.png"), src_with_labels)

# 処理時間表示
end = time.time()
print(f"time {1000.0 * (end - start):.3f}[ms]")

cv2.waitKey(0)
cv2.destroyAllWindows()
