# -*- coding: utf-8 -*-
"""
@Author  : Fang Yao
@Time    : 2023/4/1 6:07 下午
@FileName: gui.py
@desc: 字幕去除器图形化界面
"""
import os
import configparser
import PySimpleGUI as sg
import cv2
import sys
from threading import Thread
import multiprocessing
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# 把 pip 装的 nvidia 系列库（cudnn / cublas / cuda_nvrtc）路径加到 PATH
# 这样 paddlepaddle 不用手动下载 NVIDIA cuDNN 也能找到 cudnn64_8.dll
try:
    import site
    _sp_list = list(getattr(site, 'getsitepackages', lambda: [])())
    if not _sp_list and hasattr(site, 'USER_SITE') and site.USER_SITE:
        _sp_list = [site.USER_SITE]
    _candidates = list(_sp_list) + [os.path.dirname(os.path.dirname(os.__file__)) + r'\Lib\site-packages']
    _env_path = os.environ.get('PATH', '')
    for _sp in _candidates:
        _nvidia_root = os.path.join(_sp, 'nvidia')
        if not os.path.isdir(_nvidia_root):
            continue
        for _sub in ('cudnn', 'cublas', 'cuda_nvrtc', 'cuda_cupti', 'cuda_runtime', 'cufft', 'curand', 'cusolver', 'cusparse', 'nccl', 'nvtx'):
            _bin = os.path.join(_nvidia_root, _sub, 'bin')
            if os.path.isdir(_bin) and _bin not in _env_path:
                _env_path = _bin + os.pathsep + _env_path
    os.environ['PATH'] = _env_path
    # 同步加到当前进程 PATH
    if hasattr(os, 'add_dll_directory'):
        for _p in _env_path.split(os.pathsep):
            if _p and os.path.isdir(_p):
                try:
                    os.add_dll_directory(_p)
                except Exception:
                    pass
except Exception:
    pass

import backend.main
import backend.config as _vsr_config
from backend.tools.common_tools import is_image_file

# numpy 1.24+ 移除了 np.int / np.float / np.bool / np.object 别名。
# 旧版 PaddleOCR 还在用这些，做一层兼容 shim，避免 AttributeError。
try:
    import numpy as _np
    for _alias, _real in [('int', int), ('float', float), ('bool', bool), ('object', object), ('str', str), ('long', int)]:
        if not hasattr(_np, _alias):
            setattr(_np, _alias, _real)
except Exception:
    pass

# GUI 与 InpaintMode 枚举的双向映射
_MODE_TO_ENUM = {
    'sttn': _vsr_config.InpaintMode.STTN,
    'lama': _vsr_config.InpaintMode.LAMA,
    'propainter': _vsr_config.InpaintMode.PROPAINTER,
    'mosaic': _vsr_config.InpaintMode.MOSAIC,
}
_ENUM_TO_MODE_KEY = {v: k for k, v in _MODE_TO_ENUM.items()}


class SubtitleRemoverGUI:

    def __init__(self):
        # 初次运行检查运行环境是否正常
        # `paddle.fluid` was removed in PaddlePaddle 2.x; use a duck-typed
        # install check that works on both 1.x and 2.x.
        try:
            from paddle import utils as _paddle_utils  # noqa: F401
        except Exception:
            pass
        self.font = 'Arial 10'
        self.theme = 'LightBrown12'
        sg.theme(self.theme)
        self.icon = os.path.join(os.path.dirname(__file__), 'design', 'vsr.ico')
        self.screen_width, self.screen_height = sg.Window.get_screen_size()
        self.subtitle_config_file = os.path.join(os.path.dirname(__file__), 'subtitle.ini')
        print(self.screen_width, self.screen_height)
        # 设置视频预览区域大小
        self.video_preview_width = 960
        self.video_preview_height = self.video_preview_width * 9 // 16
        # 默认组件大小
        self.horizontal_slider_size = (120, 20)
        self.output_size = (100, 10)
        self.progressbar_size = (60, 20)
        # 分辨率低于1080
        if self.screen_width // 2 < 960:
            self.video_preview_width = 640
            self.video_preview_height = self.video_preview_width * 9 // 16
            self.horizontal_slider_size = (60, 20)
            self.output_size = (58, 10)
            self.progressbar_size = (28, 20)
        # 字幕提取器布局
        self.layout = None
        # 字幕提取其窗口
        self.window = None
        # 视频路径
        self.video_path = None
        # 视频cap
        self.video_cap = None
        # 视频的帧率
        self.fps = None
        # 视频的帧数
        self.frame_count = None
        # 视频的宽
        self.frame_width = None
        # 视频的高
        self.frame_height = None
        # 设置字幕区域高宽
        self.xmin = None
        self.xmax = None
        self.ymin = None
        self.ymax = None
        # 字幕提取器
        self.sr = None

    def run(self):
        # 创建布局
        self._create_layout()
        # 创建窗口
        self.window = sg.Window(title='Video Subtitle Remover', layout=self.layout,
                                icon=self.icon)
        while True:
            # 循环读取事件
            event, values = self.window.read(timeout=10)
            # 处理【算法模式切换】事件（Radio 按钮）
            self._mode_event_handler(event, values)
            # 处理【Auto-Detect】按钮
            self._auto_detect_handler(event, values)
            # 处理【打开】事件
            self._file_event_handler(event, values)
            # 处理【滑动】事件
            self._slide_event_handler(event, values)
            # 处理【运行】事件
            self._run_event_handler(event, values)
            # 如果关闭软件，退出
            if event == sg.WIN_CLOSED:
                break
            # 更新进度条
            if self.sr is not None:
                self.window['-PROG-'].update(self.sr.progress_total)
                if self.sr.preview_frame is not None:
                    self.window['-DISPLAY-'].update(data=cv2.imencode('.png', self._img_resize(self.sr.preview_frame))[1].tobytes())
                if self.sr.isFinished:
                    # 1) 打开修改字幕滑块区域按钮
                    self.window['-Y-SLIDER-'].update(disabled=False)
                    self.window['-X-SLIDER-'].update(disabled=False)
                    self.window['-Y-SLIDER-H-'].update(disabled=False)
                    self.window['-X-SLIDER-W-'].update(disabled=False)
                    # 2) 打开【运行】、【打开】和【识别语言】按钮
                    self.window['-RUN-'].update(disabled=False)
                    self.window['-FILE-'].update(disabled=False)
                    self.window['-FILE_BTN-'].update(disabled=False)
                    self.sr = None
                if len(self.video_paths) >= 1:
                    # 1) 关闭修改字幕滑块区域按钮
                    self.window['-Y-SLIDER-'].update(disabled=True)
                    self.window['-X-SLIDER-'].update(disabled=True)
                    self.window['-Y-SLIDER-H-'].update(disabled=True)
                    self.window['-X-SLIDER-W-'].update(disabled=True)
                    # 2) 关闭【运行】、【打开】和【识别语言】按钮
                    self.window['-RUN-'].update(disabled=True)
                    self.window['-FILE-'].update(disabled=True)
                    self.window['-FILE_BTN-'].update(disabled=True)

    def _mode_event_handler(self, event, values):
        """
        切换算法模式时，显示/隐藏马赛克块大小控件。
        """
        if event in ('-MODE-STTN-', '-MODE-LAMA-', '-MODE-PROP-', '-MODE-MOSAIC-'):
            is_mosaic = bool(values.get('-MODE-MOSAIC-', False))
            if self.window is not None:
                self.window['-MOSAIC-LABEL-'].update(visible=is_mosaic)
                self.window['-MOSAIC-BLOCK-'].update(visible=is_mosaic)

    def _create_layout(self):
        """
        创建字幕提取器布局
        """
        garbage = os.path.join(os.path.dirname(__file__), 'output')
        if os.path.exists(garbage):
            import shutil
            shutil.rmtree(garbage, True)
        # 当前选中的算法模式（默认沿用 config.MODE）
        default_mode_key = _ENUM_TO_MODE_KEY.get(_vsr_config.MODE, 'sttn')
        self.layout = [
            # 显示视频预览
            [sg.Image(size=(self.video_preview_width, self.video_preview_height), background_color='black',
                      key='-DISPLAY-')],
            # 打开按钮 + 快进快退条
            [sg.Input(key='-FILE-', visible=False, enable_events=True),
             sg.FilesBrowse(button_text='Open', file_types=((
                            'All Files', '*.*'), ('mp4', '*.mp4'),
                            ('flv', '*.flv'),
                            ('wmv', '*.wmv'),
                            ('avi', '*.avi')),
                            key='-FILE_BTN-', size=(10, 1), font=self.font),
             sg.Slider(size=self.horizontal_slider_size, range=(1, 1), key='-SLIDER-', orientation='h',
                       enable_events=True, font=self.font,
                       disable_number_display=True),
             ],
            # 输出区域
            [sg.Output(size=self.output_size, font=self.font),
             sg.Frame(title='Vertical', font=self.font, key='-FRAME1-',
             layout=[[
                 sg.Slider(range=(0, 0), orientation='v', size=(10, 20),
                           disable_number_display=True,
                           enable_events=True, font=self.font,
                           pad=((10, 10), (20, 20)),
                           default_value=0, key='-Y-SLIDER-'),
                 sg.Slider(range=(0, 0), orientation='v', size=(10, 20),
                           disable_number_display=True,
                           enable_events=True, font=self.font,
                           pad=((10, 10), (20, 20)),
                           default_value=0, key='-Y-SLIDER-H-'),
             ]], pad=((15, 5), (0, 0))),
             sg.Frame(title='Horizontal', font=self.font, key='-FRAME2-',
             layout=[[
                 sg.Slider(range=(0, 0), orientation='v', size=(10, 20),
                           disable_number_display=True,
                           pad=((10, 10), (20, 20)),
                           enable_events=True, font=self.font,
                           default_value=0, key='-X-SLIDER-'),
                 sg.Slider(range=(0, 0), orientation='v', size=(10, 20),
                           disable_number_display=True,
                           pad=((10, 10), (20, 20)),
                           enable_events=True, font=self.font,
                           default_value=0, key='-X-SLIDER-W-'),
             ]], pad=((15, 5), (0, 0)))
            ],

            # 自动检测水印按钮（在 Mode 行之前）
            [sg.Button(button_text='Auto-Detect Watermark', key='-AUTO-DETECT-',
                       font=self.font, size=(20, 1), disabled=True,
                       pad=((5, 5), (5, 5))),
             sg.Text('', key='-AUTO-DETECT-STATUS-', font=self.font,
                     pad=((10, 5), (5, 5))),
             ],

            # 算法模式选择（可点击的 Radio）
            [sg.Text('Mode:', font=self.font, pad=((5, 5), (8, 0))),
             sg.Radio('STTN', 'MODE', key='-MODE-STTN-',
                      default=(default_mode_key == 'sttn'),
                      font=self.font, enable_events=True),
             sg.Radio('LAMA', 'MODE', key='-MODE-LAMA-',
                      default=(default_mode_key == 'lama'),
                      font=self.font, enable_events=True),
             sg.Radio('ProPainter', 'MODE', key='-MODE-PROP-',
                      default=(default_mode_key == 'propainter'),
                      font=self.font, enable_events=True),
             sg.Radio('Mosaic', 'MODE', key='-MODE-MOSAIC-',
                      default=(default_mode_key == 'mosaic'),
                      font=self.font, enable_events=True),
             ],
            # 跳过检测勾选 + 马赛克块大小
            [sg.Checkbox('Skip detection (use sub_area as mask — works for STTN & LAMA)', key='-SKIP-DET-',
                         default=bool(_vsr_config.STTN_SKIP_DETECTION),
                         font=self.font, pad=((5, 5), (0, 0))),
             sg.Text('Mosaic block (px):', font=self.font, pad=((15, 3), (0, 0)),
                     key='-MOSAIC-LABEL-', visible=(default_mode_key == 'mosaic')),
             sg.Slider(range=(4, 80), default_value=int(getattr(_vsr_config, 'MOSAIC_BLOCK_SIZE', 20)),
                       resolution=1, orientation='h', size=(20, 20),
                       key='-MOSAIC-BLOCK-', font=self.font, enable_events=True,
                       visible=(default_mode_key == 'mosaic')),
             ],

            # 运行按钮 + 进度条
            [sg.Button(button_text='Run', key='-RUN-',
                       font=self.font, size=(20, 1)),
             sg.ProgressBar(100, orientation='h', size=self.progressbar_size, key='-PROG-', auto_size_text=True)
             ],
        ]

    def _file_event_handler(self, event, values):
        """
        当点击打开按钮时：
        1）打开视频文件，将画布显示视频帧
        2）获取视频信息，初始化进度条滑块范围
        """
        if event == '-FILE-':
            self.video_paths = values['-FILE-'].split(';')
            self.video_path = self.video_paths[0]
            if self.video_path != '':
                self.video_cap = cv2.VideoCapture(self.video_path)
            if self.video_cap is None:
                return
            if self.video_cap.isOpened():
                ret, frame = self.video_cap.read()
                if ret:
                    for video in self.video_paths:
                        print(f"Open Video Success：{video}")
                    # 获取视频的帧数
                    self.frame_count = self.video_cap.get(cv2.CAP_PROP_FRAME_COUNT)
                    # 获取视频的高度
                    self.frame_height = self.video_cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
                    # 获取视频的宽度
                    self.frame_width = self.video_cap.get(cv2.CAP_PROP_FRAME_WIDTH)
                    # 获取视频的帧率
                    self.fps = self.video_cap.get(cv2.CAP_PROP_FPS)
                    # 调整视频帧大小，使播放器能够显示
                    resized_frame = self._img_resize(frame)
                    # resized_frame = cv2.resize(src=frame, dsize=(self.video_preview_width, self.video_preview_height))
                    # 显示视频帧
                    self.window['-DISPLAY-'].update(data=cv2.imencode('.png', resized_frame)[1].tobytes())
                    # 更新视频进度条滑块range
                    self.window['-SLIDER-'].update(range=(1, self.frame_count))
                    self.window['-SLIDER-'].update(1)
                    # 预设字幕区域位置
                    y_p, h_p, x_p, w_p = self.parse_subtitle_config()
                    y = self.frame_height * y_p
                    h = self.frame_height * h_p
                    x = self.frame_width * x_p
                    w = self.frame_width * w_p
                    # 更新视频字幕位置滑块range
                    # 更新Y-SLIDER范围
                    self.window['-Y-SLIDER-'].update(range=(0, self.frame_height), disabled=False)
                    # 更新Y-SLIDER默认值
                    self.window['-Y-SLIDER-'].update(y)
                    # 更新X-SLIDER范围
                    self.window['-X-SLIDER-'].update(range=(0, self.frame_width), disabled=False)
                    # 更新X-SLIDER默认值
                    self.window['-X-SLIDER-'].update(x)
                    # 更新Y-SLIDER-H范围
                    self.window['-Y-SLIDER-H-'].update(range=(0, self.frame_height - y))
                    # 更新Y-SLIDER-H默认值
                    self.window['-Y-SLIDER-H-'].update(h)
                    # 更新X-SLIDER-W范围
                    self.window['-X-SLIDER-W-'].update(range=(0, self.frame_width - x))
                    # 更新X-SLIDER-W默认值
                    self.window['-X-SLIDER-W-'].update(w)
                    self._update_preview(frame, (y, h, x, w))
                    # 启用 Auto-Detect 按钮
                    self.window['-AUTO-DETECT-'].update(disabled=False)

    def _auto_detect_handler(self, event, values):
        """
        Auto-Detect Watermark 按钮：扫第一帧右下角，自动定位 ✦ 位置。
        适用 LAMA / STTN + 跳过检测模式。
        """
        if event != '-AUTO-DETECT-':
            return
        if self.video_cap is None or not self.video_paths:
            print('[Auto-Detect] 请先 Open 视频')
            return
        try:
            from remove_watermark import detect_bottom_right_watermark
            video_path = self.video_paths[0]
            self.window['-AUTO-DETECT-STATUS-'].update('Detecting...')
            self.window.refresh() if hasattr(self.window, 'refresh') else None
            sub_area = detect_bottom_right_watermark(video_path, padding=20)
            if sub_area is None:
                print('[Auto-Detect] 未找到水印，请手动拖动滑块')
                self.window['-AUTO-DETECT-STATUS-'].update('Not found, drag manually')
                return
            ymin, ymax, xmin, xmax = sub_area
            # 更新滑块
            self.window['-Y-SLIDER-'].update(ymin)
            self.window['-Y-SLIDER-H-'].update(ymax - ymin)
            self.window['-X-SLIDER-'].update(xmin)
            self.window['-X-SLIDER-W-'].update(xmax - xmin)
            # 自动切到 STTN + 跳过检测（最适合半透明图形水印：不"生成"内容，
            # 而是从其他帧复制最像的 patch，无变形）
            self.window['-MODE-STTN-'].update(True)
            self.window['-SKIP-DET-'].update(True)
            # 显示状态
            msg = f'Auto-detected at ({xmin},{ymin})~({xmax},{ymax}) | STTN + skip ON'
            print(f'[Auto-Detect] {msg}')
            self.window['-AUTO-DETECT-STATUS-'].update(msg)
            # 更新预览
            cap = cv2.VideoCapture(video_path)
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            ret, frame = cap.read()
            cap.release()
            if ret:
                self._update_preview(frame, (ymin, ymax - ymin, xmin, xmax - xmin))
        except Exception as e:
            print(f'[Auto-Detect] 失败: {e}')
            self.window['-AUTO-DETECT-STATUS-'].update(f'Failed: {e}')

    def __disable_button(self):
        # 1) 禁止修改字幕滑块区域
        self.window['-Y-SLIDER-'].update(disabled=True)
        self.window['-X-SLIDER-'].update(disabled=True)
        self.window['-Y-SLIDER-H-'].update(disabled=True)
        self.window['-X-SLIDER-W-'].update(disabled=True)
        # 2) 禁止再次点击【运行】、【打开】和【识别语言】按钮
        self.window['-RUN-'].update(disabled=True)
        self.window['-FILE-'].update(disabled=True)
        self.window['-FILE_BTN-'].update(disabled=True)

    def _run_event_handler(self, event, values):
        """
        当点击运行按钮时：
        1) 禁止修改字幕滑块区域
        2) 禁止再次点击【运行】和【打开】按钮
        3) 设定字幕区域位置
        """
        if event == '-RUN-':
            if self.video_cap is None:
                print('Please Open Video First')
            else:
                # 禁用按钮
                self.__disable_button()
                # 3) 设定字幕区域位置
                self.xmin = int(values['-X-SLIDER-'])
                self.xmax = int(values['-X-SLIDER-'] + values['-X-SLIDER-W-'])
                self.ymin = int(values['-Y-SLIDER-'])
                self.ymax = int(values['-Y-SLIDER-'] + values['-Y-SLIDER-H-'])
                if self.ymax > self.frame_height:
                    self.ymax = self.frame_height
                if self.xmax > self.frame_width:
                    self.xmax = self.frame_width
                if len(self.video_paths) <= 1:
                    subtitle_area = (self.ymin, self.ymax, self.xmin, self.xmax)
                else:
                    print(f"{'Processing multiple videos or images'}")
                    # 先判断每个视频的分辨率是否一致，一致的话设置相同的字幕区域，否则设置为None
                    global_size = None
                    for temp_video_path in self.video_paths:
                        temp_cap = cv2.VideoCapture(temp_video_path)
                        if global_size is None:
                            global_size = (int(temp_cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(temp_cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
                        else:
                            temp_size = (int(temp_cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(temp_cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
                            if temp_size != global_size:
                                print('not all video/images in same size, processing in full screen')
                                subtitle_area = None
                    else:
                        subtitle_area = (self.ymin, self.ymax, self.xmin, self.xmax)
                y_p = self.ymin / self.frame_height
                h_p = (self.ymax - self.ymin) / self.frame_height
                x_p = self.xmin / self.frame_width
                w_p = (self.xmax - self.xmin) / self.frame_width
                self.set_subtitle_config(y_p, h_p, x_p, w_p)

                # 读取 GUI 上选择的算法模式 / 跳过检测 / 马赛克块大小
                run_mode_key = self._selected_mode_key(values)
                run_mode_enum = _MODE_TO_ENUM[run_mode_key]
                skip_detection = bool(values.get('-SKIP-DET-', False))
                mosaic_block = int(values.get('-MOSAIC-BLOCK-', getattr(_vsr_config, 'MOSAIC_BLOCK_SIZE', 20)))

                def task():
                    while self.video_paths:
                        video_path = self.video_paths.pop()
                        if subtitle_area is not None:
                            print(f"{'SubtitleArea'}：({self.ymin},{self.ymax},{self.xmin},{self.xmax})")
                        print(f"[Mode] {run_mode_key} | skip_detection={skip_detection} | mosaic_block={mosaic_block}px")
                        self.sr = backend.main.SubtitleRemover(
                            video_path,
                            subtitle_area,
                            True,
                            mode=run_mode_enum,
                            skip_detection=skip_detection,
                            mosaic_block_size=mosaic_block,
                        )
                        self.__disable_button()
                        self.sr.run()
                Thread(target=task, daemon=True).start()
                self.video_cap.release()
                self.video_cap = None

    @staticmethod
    def _selected_mode_key(values):
        """
        从 PySimpleGUI 的 values 字典里读出当前选中的算法模式 key。
        Radio 共享同一个 group_id='MODE'，因此同一时刻只有一个 key 为 True。
        """
        for key in ('-MODE-STTN-', '-MODE-LAMA-', '-MODE-PROP-', '-MODE-MOSAIC-'):
            if values.get(key, False):
                return {
                    '-MODE-STTN-': 'sttn',
                    '-MODE-LAMA-': 'lama',
                    '-MODE-PROP-': 'propainter',
                    '-MODE-MOSAIC-': 'mosaic',
                }[key]
        return 'sttn'

    def _slide_event_handler(self, event, values):
        """
        当滑动视频进度条/滑动字幕选择区域滑块时：
        1) 判断视频是否存在，如果存在则显示对应的视频帧
        2) 绘制rectangle
        """
        if event == '-SLIDER-' or event == '-Y-SLIDER-' or event == '-Y-SLIDER-H-' or event == '-X-SLIDER-' or event \
                == '-X-SLIDER-W-':
            # 判断是否时单张图片
            if is_image_file(self.video_path):
                img = cv2.imread(self.video_path)
                self.window['-Y-SLIDER-H-'].update(range=(0, self.frame_height - values['-Y-SLIDER-']))
                self.window['-X-SLIDER-W-'].update(range=(0, self.frame_width - values['-X-SLIDER-']))
                # 画字幕框
                y = int(values['-Y-SLIDER-'])
                h = int(values['-Y-SLIDER-H-'])
                x = int(values['-X-SLIDER-'])
                w = int(values['-X-SLIDER-W-'])
                self._update_preview(img, (y, h, x, w))
            elif self.video_cap is not None and self.video_cap.isOpened():
                frame_no = int(values['-SLIDER-'])
                self.video_cap.set(cv2.CAP_PROP_POS_FRAMES, frame_no)
                ret, frame = self.video_cap.read()
                if ret:
                    self.window['-Y-SLIDER-H-'].update(range=(0, self.frame_height-values['-Y-SLIDER-']))
                    self.window['-X-SLIDER-W-'].update(range=(0, self.frame_width-values['-X-SLIDER-']))
                    # 画字幕框
                    y = int(values['-Y-SLIDER-'])
                    h = int(values['-Y-SLIDER-H-'])
                    x = int(values['-X-SLIDER-'])
                    w = int(values['-X-SLIDER-W-'])
                    self._update_preview(frame, (y, h, x, w))

    def _update_preview(self, frame, y_h_x_w):
        y, h, x, w = y_h_x_w
        # 画字幕框
        draw = cv2.rectangle(img=frame, pt1=(int(x), int(y)), pt2=(int(x) + int(w), int(y) + int(h)),
                             color=(0, 255, 0), thickness=3)
        # 调整视频帧大小，使播放器能够显示
        resized_frame = self._img_resize(draw)
        # 显示视频帧
        self.window['-DISPLAY-'].update(data=cv2.imencode('.png', resized_frame)[1].tobytes())

    def _img_resize(self, image):
        top, bottom, left, right = (0, 0, 0, 0)
        height, width = image.shape[0], image.shape[1]
        # 对长短不想等的图片，找到最长的一边
        longest_edge = height
        # 计算短边需要增加多少像素宽度使其与长边等长
        if width < longest_edge:
            dw = longest_edge - width
            left = dw // 2
            right = dw - left
        else:
            pass
        # 给图像增加边界
        constant = cv2.copyMakeBorder(image, top, bottom, left, right, cv2.BORDER_CONSTANT, value=[0, 0, 0])
        return cv2.resize(constant, (self.video_preview_width, self.video_preview_height))

    def set_subtitle_config(self, y, h, x, w):
        # 写入配置文件
        with open(self.subtitle_config_file, mode='w', encoding='utf-8') as f:
            f.write('[AREA]\n')
            f.write(f'Y = {y}\n')
            f.write(f'H = {h}\n')
            f.write(f'X = {x}\n')
            f.write(f'W = {w}\n')

    def parse_subtitle_config(self):
        y_p, h_p, x_p, w_p = .78, .21, .05, .9
        # 如果配置文件不存在，则写入配置文件
        if not os.path.exists(self.subtitle_config_file):
            self.set_subtitle_config(y_p, h_p, x_p, w_p)
            return y_p, h_p, x_p, w_p
        else:
            try:
                config = configparser.ConfigParser()
                config.read(self.subtitle_config_file, encoding='utf-8')
                conf_y_p, conf_h_p, conf_x_p, conf_w_p = float(config['AREA']['Y']), float(config['AREA']['H']), float(config['AREA']['X']), float(config['AREA']['W'])
                return conf_y_p, conf_h_p, conf_x_p, conf_w_p
            except Exception:
                self.set_subtitle_config(y_p, h_p, x_p, w_p)
                return y_p, h_p, x_p, w_p


if __name__ == '__main__':
    try:
        multiprocessing.set_start_method("spawn")
        # 运行图形化界面
        subtitleRemoverGUI = SubtitleRemoverGUI()
        subtitleRemoverGUI.run()
    except Exception as e:
        print(f'[{type(e)}] {e}')
        import traceback
        traceback.print_exc()
        msg = traceback.format_exc()
        err_log_path = os.path.join(os.path.expanduser('~'), 'VSR-Error-Message.log')
        with open(err_log_path, 'w', encoding='utf-8') as f:
            f.writelines(msg)
        import platform
        if platform.system() == 'Windows':
            os.system('pause')
        else:
            input()
