"""Test STTN with tuned params on user's video."""
import sys, os
sys.path.insert(0, r'D:\BaiduNetdiskWorkspace\Py-soft\video-subtitle-remover')
# 触发 PATH + numpy shim
import gui  # noqa
import backend.config as c
# 临时清空 sys.argv 避免 LAMA/OCR argparse 报错
_orig_argv = sys.argv[:]
sys.argv = [sys.argv[0]]
try:
    from backend.main import SubtitleRemover
    from remove_watermark import detect_bottom_right_watermark
finally:
    sys.argv = _orig_argv

video_path = r'C:\Users\Administrator\Downloads\按照脚本，生成前_个镜头，_人物一致性要强，_真实质感，角色.mp4'

print('[1/2] 检测水印位置...')
sub_area = detect_bottom_right_watermark(video_path, padding=20)
print(f'  sub_area = {sub_area}')

print('[2/2] STTN inpaint...')
sr = SubtitleRemover(
    video_path,
    sub_area=sub_area,
    gui_mode=False,
    mode=c.InpaintMode.STTN,
    skip_detection=True,
)
sr.run()
print(f'\n✓ 完成: {sr.video_out_name}')
