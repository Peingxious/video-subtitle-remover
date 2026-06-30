"""Compare first/last frame of the new LAMA output to original."""
import cv2
paths = [
    (r'C:\Users\Administrator\Downloads\按照脚本，生成前_个镜头，_人物一致性要强，_真实质感，角色.mp4', 'orig'),
    (r'C:\Users\Administrator\Downloads\按照脚本，生成前_个镜头，_人物一致性要强，_真实质感，角色_no_sub.mp4', 'lama'),
]
for path, name in paths:
    cap = cv2.VideoCapture(path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    w, h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    for label, idx in [('first', 0), ('mid', total//2), ('last', total-1)]:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if not ret:
            continue
        br = frame[h//2:, w//2:]
        cv2.imwrite(f'_br2_{name}_{label}.png', br)
    cap.release()
print('done')
