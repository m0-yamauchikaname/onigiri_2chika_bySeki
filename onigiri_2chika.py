import cv2
import numpy as np
import time
import os

# タイマー開始
start = time.time()

# --- 0. 入力画像読み込み ---
input_folder_name= "input_images\sample1.jpg"
script_dir = os.path.dirname(os.path.abspath(__file__))
folder_path = os.path.join(script_dir, input_folder_name)
src_full = cv2.imread(folder_path, cv2.IMREAD_COLOR)
if src_full is None:
    print("画像が見つかりません。")
    exit(-1)

# 中心からトリミング
center_x = src_full.shape[1] // 2
center_y = src_full.shape[0] // 2
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
light_src = cv2.convertScaleAbs(src, alpha=alpha, beta=beta)

# グレースケール変換
gray = cv2.cvtColor(light_src, cv2.COLOR_BGR2GRAY)

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

# --- 5. ラベリング ---
nLabels, labels = cv2.connectedComponents(morph_close)

# --- 6. 赤色を含むラベルだけ抽出 ---
# ラベルのピクセル数の閾値を設定
pixel_threshold_min = 10000
pixel_threshold_max = 49650
filtered = np.zeros_like(labels, dtype=np.uint8)
for i in range(1, nLabels):
    mask = (labels == i).astype(np.uint8)
    overlap = cv2.bitwise_and(red_mask, red_mask, mask=mask)
    if cv2.countNonZero(overlap) > 0:
        if cv2.countNonZero(mask) > pixel_threshold_min^2 and cv2.countNonZero(mask) < pixel_threshold_max^2:
            filtered[labels == i] = 255



# --- ラベリングし直し＆ID表示 ---
nLabels_filtered, labels_filtered = cv2.connectedComponents(filtered)

label_display = cv2.cvtColor(filtered, cv2.COLOR_GRAY2BGR)
src_with_labels = src.copy()

for i in range(1, nLabels_filtered):
    mask = (labels_filtered == i).astype(np.uint8)
    m = cv2.moments(mask)
    if m["m00"] != 0:
        cx = int(m["m10"] / m["m00"])
        cy = int(m["m01"] / m["m00"])
        cv2.putText(label_display, "*", (cx, cy), cv2.FONT_HERSHEY_SIMPLEX, 3.0, (0, 0, 255), 2)
        cv2.putText(label_display, str(i), (cx + 50, cy), cv2.FONT_HERSHEY_SIMPLEX, 3.0, (0, 0, 255), 10)
        cv2.putText(src_with_labels, "*", (cx, cy), cv2.FONT_HERSHEY_SIMPLEX, 3.0, (0, 255, 0), 2)
        cv2.putText(src_with_labels, str(i), (cx + 50, cy), cv2.FONT_HERSHEY_SIMPLEX, 3.0, (0, 0, 255), 10)

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
show_resized("Lightened", light_src, 0)
show_resized("Gray", gray, 1)
show_resized("Otsu Binary", otsu_binary, 2)
show_resized("Morph Open", morph_open, 3)
show_resized("Morph Close", morph_close, 4)
show_resized("Red Mask (HSV)", red_mask, 5)
show_resized("Filtered Labels", filtered, 6)
show_resized("Labels with ID", label_display, 7)
show_resized("Original with Labels", src_with_labels, 8)

# --- 画像保存 ---
# 保存
folder_path = os.path.join(script_dir, "output_images")
cv2.imwrite(os.path.join(folder_path, "output_original_cropped.png"), src)
cv2.imwrite(os.path.join(folder_path, "output_lightened.png"), light_src)
cv2.imwrite(os.path.join(folder_path, "output_gray.png"), gray)
cv2.imwrite(os.path.join(folder_path, "output_otsu_binary.png"), otsu_binary)
cv2.imwrite(os.path.join(folder_path, "output_morph_open.png"), morph_open)
cv2.imwrite(os.path.join(folder_path, "output_morph_close.png"), morph_close)
cv2.imwrite(os.path.join(folder_path, "output_red_mask.png"), red_mask)
cv2.imwrite(os.path.join(folder_path, "output_filtered_labels.png"), filtered)
cv2.imwrite(os.path.join(folder_path, "output_labels_with_id.png"), label_display)

# 処理時間表示
end = time.time()
print(f"time {1000.0 * (end - start):.3f}[ms]")

cv2.waitKey(0)
cv2.destroyAllWindows()
