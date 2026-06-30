"""Compare original and processed videos, focus on the bottom-right corner."""
import cv2
import os

paths = [
    (r'C:\Users\Administrator\Downloads\按照脚本，生成前_个镜头，_人物一致性要强，_真实质感，角色.mp4', 'original'),
    (r'C:\Users\Administrator\Downloads\按照脚本，生成前_个镜头，_人物一致性要强，_真实质感，角色_no_sub.mp4', 'processed'),
]

# 取第一帧和最后一帧，对比右下角 1/4 区域
for path, name in paths:
    if not os.path.exists(path):
        print(f'[missing] {path}')
        continue
    cap = cv2.VideoCapture(path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    print(f'\n=== {name} ===')
    print(f'  path={path}')
    print(f'  size={w}x{h}, total_frames={total}, fps={fps:.2f}')

    for label, idx in [('first', 0), ('last', total-1)]:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if not ret:
            continue
        # 右下角 1/4
        br = frame[h//2:, w//2:]
        out_p = f'_br_{name}_{label}.png'
        cv2.imwrite(out_p, br)
        print(f'  {label} frame idx={idx}: bottom-right saved to {out_p}, shape={br.shape}')
    cap.release()
print('\n=== done ===')
