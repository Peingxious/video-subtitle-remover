"""Test LAMA mode with skip detection on the user's actual video."""
import sys, os
sys.path.insert(0, r'D:\BaiduNetdiskWorkspace\Py-soft\video-subtitle-remover')
import gui  # 触发 PATH + numpy shim

import cv2, numpy as np
import backend.config as c
from backend.main import SubtitleRemover

video_path = r'C:\Users\Administrator\Downloads\按照脚本，生成前_个镜头，_人物一致性要强，_真实质感，角色.mp4'

# 1. 探一下原视频大小 + 水印大致位置
cap = cv2.VideoCapture(video_path)
w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
print(f'原视频: {w}x{h}')

# 取 5 帧，算右下角亮点的 bounding box
bright_regions = []
for i in [0, 30, 60, 90, 120]:
    cap.set(cv2.CAP_PROP_POS_FRAMES, i)
    ret, frame = cap.read()
    if not ret:
        continue
    # 右下角 1/4 区域
    br = frame[h//2:, w//2:]
    gray = cv2.cvtColor(br, cv2.COLOR_BGR2GRAY)
    # 水印是亮色，阈值化
    _, mask = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY)
    # 找白色像素的 bbox
    ys, xs = np.where(mask > 0)
    if len(xs) > 5 and len(ys) > 5:
        # 转回原图坐标
        xmin = xs.min() + w//2
        xmax = xs.max() + w//2
        ymin = ys.min() + h//2
        ymax = ys.max() + h//2
        bright_regions.append((xmin, ymin, xmax, ymax))
        print(f'  frame {i}: detected bright region ({xmin},{ymin})~({xmax},{ymax}) = {xmax-xmin}x{ymax-ymin}px')
cap.release()

# 2. 用所有帧的并集作为 sub_area（包一圈）
# 智能检测：找右下角 1/4 区域里、形态接近方形的亮团
if bright_regions:
    # 先用 OpenCV 的轮廓筛选
    cap = cv2.VideoCapture(video_path)
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    ret, frame = cap.read()
    cap.release()
    # 取右下 1/4，转灰度
    br = frame[h//2:, w//2:]
    gray = cv2.cvtColor(br, cv2.COLOR_BGR2GRAY)
    # 二值化（亮区域）
    _, bw = cv2.threshold(gray, 220, 255, cv2.THRESH_BINARY)
    # 找连通区域
    n, labels, stats, _ = cv2.connectedComponentsWithStats(bw, connectivity=8)
    # 找最小的、面积 50-2000 之间的连通区域（水印大小）
    candidates = []
    for i in range(1, n):  # 跳过背景
        area = stats[i, cv2.CC_STAT_AREA]
        ww = stats[i, cv2.CC_STAT_WIDTH]
        hh = stats[i, cv2.CC_STAT_HEIGHT]
        if 30 < area < 5000 and 0.5 < ww/hh < 2.0 and ww < 200 and hh < 200:
            x = stats[i, cv2.CC_STAT_LEFT]
            y = stats[i, cv2.CC_STAT_TOP]
            candidates.append((x + w//2, y + h//2, x + w//2 + ww, y + h//2 + hh, area))
    if candidates:
        # 取最右下的
        candidates.sort(key=lambda c: -(c[2] + c[3]))  # xmax+ymax 最大
        xmin, ymin, xmax, ymax, _ = candidates[0]
        # 留 20px 边距
        sub_area = (max(0, ymin - 20), min(h, ymax + 20), max(0, xmin - 20), min(w, xmax + 20))
        print(f'  detected: ({xmin},{ymin})~({xmax},{ymax}) = {xmax-xmin}x{ymax-ymin}px')
        print(f'  with 20px padding -> sub_area: {sub_area}')
    else:
        # 退化：用右下 200x200
        sub_area = (h - 200, h, w - 200, w)
        print(f'  no candidate found, fallback to bottom-right 200x200: {sub_area}')

# 3. 跑 LAMA + skip detection
print(f'\n=== LAMA + skip detection on {os.path.basename(video_path)} ===')
sr = SubtitleRemover(
    video_path,
    sub_area=sub_area,
    gui_mode=False,
    mode=c.InpaintMode.LAMA,
    skip_detection=True,
)
sr.run()
out = sr.video_out_name
print(f'output: {out}, size: {os.path.getsize(out) if os.path.exists(out) else 0} bytes')
