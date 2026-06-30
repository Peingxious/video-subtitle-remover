"""
智能水印检测 + LAMA inpaint — 一键去除 AI 视频右下角的水印（✦/台标等）。

用法：
    python remove_watermark.py 视频路径.mp4

算法：
    1. 读第一帧，找右下角 1/4 区域里形态接近方形、面积 30-5000 px 的最亮连通区域
    2. 加 20px padding 得到 sub_area
    3. LAMA 模式 + 跳过 OCR 检测，对全帧做 inpaint
    4. 输出到 同目录/<原名>_cleaned.mp4
"""
import sys
import os
import cv2
import numpy as np


def detect_bottom_right_watermark(video_path, padding=20, debug=False):
    """
    检测视频右下角的水印位置。
    返回 (ymin, ymax, xmin, xmax)，失败时返回 None。
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return None
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    ret, frame = cap.read()
    cap.release()
    if not ret:
        return None

    # 扫描多个帧的"稳定"水印（位置变化幅度小的连通区域）
    cap = cv2.VideoCapture(video_path)
    n_frames = min(int(cap.get(cv2.CAP_PROP_FRAME_COUNT)), 8)
    detections = []
    for i in np.linspace(0, n_frames - 1, 5, dtype=int):
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(i))
        ret, fr = cap.read()
        if not ret:
            continue
        br = fr[h // 2:, w // 2:]
        gray = cv2.cvtColor(br, cv2.COLOR_BGR2GRAY)
        # 亮区域
        _, bw = cv2.threshold(gray, 220, 255, cv2.THRESH_BINARY)
        n, _, stats, _ = cv2.connectedComponentsWithStats(bw, connectivity=8)
        for j in range(1, n):
            area = stats[j, cv2.CC_STAT_AREA]
            ww = stats[j, cv2.CC_STAT_WIDTH]
            hh = stats[j, cv2.CC_STAT_HEIGHT]
            # 候选条件：面积适中、近似方形、尺寸 < 200px
            if 30 < area < 5000 and 0.4 < ww / hh < 2.5 and 10 < ww < 200 and 10 < hh < 200:
                x = stats[j, cv2.CC_STAT_LEFT] + w // 2
                y = stats[j, cv2.CC_STAT_TOP] + h // 2
                detections.append((x, y, x + ww, y + hh))
    cap.release()

    if not detections:
        return None
    # 投票：取出现 ≥ 3 次的位置（稳定）
    from collections import Counter
    counter = Counter()
    for d in detections:
        # 量化为 20px 网格
        qx, qy = d[0] // 20, d[1] // 20
        counter[(qx, qy)] += 1
    if not counter:
        return None
    best = counter.most_common(1)[0][0]
    # 用该位置的所有检测求 bbox
    matches = [d for d in detections if (d[0] // 20, d[1] // 20) == best]
    xs = [m[0] for m in matches] + [m[2] for m in matches]
    ys = [m[1] for m in matches] + [m[3] for m in matches]
    xmin = max(0, min(xs) - padding)
    xmax = min(w, max(xs) + padding)
    ymin = max(0, min(ys) - padding)
    ymax = min(h, max(ys) + padding)
    if debug:
        print(f'  detected bbox: ({xmin},{ymin})~({xmax},{ymax}) = {xmax-xmin}x{ymax-ymin}px')
        print(f'  votes: {counter.most_common(3)}')
    return (ymin, ymax, xmin, xmax)


def main():
    if len(sys.argv) < 2:
        print('用法: python remove_watermark.py 视频路径.mp4 [输出.mp4]')
        sys.exit(1)
    video_path = sys.argv[1]
    if not os.path.exists(video_path):
        print(f'[错误] 文件不存在: {video_path}')
        sys.exit(1)

    print(f'[1/3] 检测水印位置...')
    sub_area = detect_bottom_right_watermark(video_path, debug=True)
    if sub_area is None:
        print('  [警告] 没找到合适的水印区域，退出')
        sys.exit(2)
    print(f'  sub_area = {sub_area}')

    print(f'[2/3] 加载模型...')
    # 触发 PATH + numpy shim；同时把 sys.argv 暂时清空
    # 否则 LAMA/PaddleOCR 的 argparse 会把视频路径当成未知参数
    import gui  # noqa
    import backend.config as c
    # 临时清空 sys.argv（保留 [0]），避免内部 argparse 报错
    _orig_argv = sys.argv[:]
    sys.argv = [sys.argv[0]]
    try:
        from backend.main import SubtitleRemover
        total = int(__import__('cv2').VideoCapture(video_path).get(cv2.CAP_PROP_FRAME_COUNT))
        # STTN 对半透明水印最稳：复制其他帧的无水印区域而不是"生成"内容
        # LAMA 太重（生成式 inpaint，色温/纹理会偏移）
        # ProPainter 显存太大
        # Mosaic 是有损的（马赛克）
        mode = c.InpaintMode.STTN
        print(f'[3/3] {mode.value} inpaint 全部 {total} 帧（stride={c.STTN_NEIGHBOR_STRIDE}, ref_len={c.STTN_REFERENCE_LENGTH}）...')
        sr = SubtitleRemover(
            video_path,
            sub_area=sub_area,
            gui_mode=False,
            mode=mode,
            skip_detection=True,
        )
        sr.run()
    finally:
        sys.argv = _orig_argv
    print(f'\n✓ 完成: {sr.video_out_name}')


if __name__ == '__main__':
    main()
