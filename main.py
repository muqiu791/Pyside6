import os
import re
import traceback

from ultralytics.engine.predictor import BasePredictor
from ultralytics.engine.results import Results
from ultralytics.utils import DEFAULT_CFG, LOGGER, SETTINGS, callbacks, ops
from ultralytics.utils.plotting import Annotator, colors, save_one_box
from ultralytics.utils.torch_utils import smart_inference_mode
from ultralytics.utils.files import increment_path
from ultralytics.utils.checks import check_imshow
from ultralytics.cfg import get_cfg

# from ultralytics.yolo.engine.predictor import BasePredictor
# from ultralytics.yolo.engine.results import Results
# from ultralytics.yolo.utils import DEFAULT_CFG, LOGGER, SETTINGS, callbacks, ops
# from ultralytics.yolo.utils.plotting import Annotator, colors, save_one_box
# from ultralytics.yolo.utils.torch_utils import smart_inference_mode
# from ultralytics.yolo.utils.files import increment_path
# from ultralytics.yolo.utils.checks import check_imshow
# from ultralytics.yolo.cfg import get_cfg

from PySide6.QtWidgets import QApplication, QMainWindow, QFileDialog, QMenu
from PySide6.QtGui import QImage, QPixmap, QColor,QCursor
from PySide6.QtCore import QTimer, QThread, Signal, QObject, QPoint, Qt
from ui.CustomMessageBox import MessageBox
from ui.home import Ui_MainWindow
from ui import resources_rc
from UIFunctions import *
from collections import defaultdict
from pathlib import Path
from utils.capnums import Camera
from utils.rtsp_win import Window
import numpy as np
import time
import json
import torch
import sys
import cv2

import supervision as sv
from ultralytics import YOLO
from ultralytics.data.loaders import LoadStreams

from ultralytics.engine.predictor import BasePredictor
from ultralytics.utils import DEFAULT_CFG, SETTINGS
from ultralytics.utils.torch_utils import smart_inference_mode
from ultralytics.utils.files import increment_path
from ultralytics.cfg import get_cfg
from ultralytics.utils.checks import check_imshow

from PySide6.QtCore import Signal, QObject, Qt
from PySide6.QtWidgets import QDialog, QVBoxLayout, QLabel, QCheckBox, QPushButton, QMessageBox

from PIL import Image
from pathlib import Path
import datetime
import numpy as np
import time
import cv2
import onnxruntime as ort

import geopandas as gpd
import matplotlib.pyplot as plt

x_axis_time_graph = []
y_axis_count_graph = []
video_id_count = 0


class YoloPredictor(BasePredictor, QObject):
    yolo2main_trail_img = Signal(np.ndarray)  # 轨迹图像信号
    yolo2main_box_img = Signal(np.ndarray)  # 绘制了标签与锚框的图像的信号
    yolo2main_status_msg = Signal(str)  # 检测/暂停/停止/测试完成等信号
    yolo2main_fps = Signal(str)  # fps

    yolo2main_labels = Signal(dict)  # 检测到的目标结果（每个类别的数量）
    yolo2main_progress = Signal(int)  # 进度条
    yolo2main_class_num = Signal(int)  # 当前帧类别数
    yolo2main_target_num = Signal(int)  # 当前帧目标数
    non_good_detected = Signal()

    def __init__(self, cfg=DEFAULT_CFG, overrides=None):
        super(YoloPredictor, self).__init__()
        QObject.__init__(self)

        try:
            self.args = get_cfg(cfg, overrides)
        except:
            pass
        project = self.args.project or Path(SETTINGS['runs_dir']) / self.args.task
        name = f'{self.args.mode}'
        self.save_dir = increment_path(Path(project) / name, exist_ok=self.args.exist_ok)
        self.done_warmup = False
        if self.args.show:
            self.args.show = check_imshow(warn=True)

        # GUI args
        self.used_model_name = None  # 使用过的检测模型名称
        self.new_model_name = None  # 新更改的模型

        self.source = ''  # 输入源str
        self.progress_value = 0  # 进度条的值

        self.stop_dtc = False  # 终止bool
        self.continue_dtc = True  # 暂停bool


        # config
        self.iou_thres = 0.45  # iou
        self.conf_thres = 0.25  # conf
        self.speed_thres = 0.01  # delay, ms （缓冲）

        self.save_res = False  # 保存MP4
        self.save_txt = False  # 保存txt
        self.save_res_path = "pre_result"
        self.save_txt_path = "pre_labels"

        self.show_labels = True  # 显示图像标签bool
        self.show_trace = False  # 显示图像轨迹bool


        # 运行时候的参数放这里
        self.start_time = None  # 拿来算FPS的计数变量
        self.count = None
        self.sum_of_count = None
        self.class_num = None
        self.total_frames = None
        self.lock_id = None

        # 设置线条样式    厚度 & 缩放大小
        self.box_annotator = sv.BoxAnnotator(
            thickness=2,  # 增加边界框和文本的厚度
            text_thickness=1,  # 增加文本的厚度
            text_scale=0.5,  # 调整文本的缩放比例
        )

        self.ort_session = None

    # 点击开始检测按钮后的检测事件
    @smart_inference_mode()  # 一个修饰器，用来开启检测模式：如果torch>=1.9.0，则执行torch.inference_mode()，否则执行torch.no_grad()
    def run(self):
        self.yolo2main_status_msg.emit('正在加载模型...')
        LoadStreams.capture = ''
        self.count = 0                 # 拿来参与算FPS的计数变量
        self.start_time = time.time()  # 拿来算FPS的计数变量
        global video_id_count

        # 检查保存路径
        if self.save_txt:
            self.check_path(self.save_txt_path)
        if self.save_res:
            self.check_path(self.save_res_path)

        self.yolo2main_status_msg.emit('正在加载模型...')
        model=self.load_yolo_model()

        # 获取数据源 （不同的类型获取不同的数据源）
        iter_model = iter(
            model.track(source=self.source, show=False, stream=True, iou=self.iou_thres, conf=self.conf_thres))

        # 折线图数据初始化
        global x_axis_time_graph, y_axis_count_graph
        x_axis_time_graph = []
        y_axis_count_graph = []

        self.yolo2main_status_msg.emit('检测中...')

        # 使用OpenCV读取视频——获取进度条
        if 'mp4' in self.source or 'avi' in self.source or 'mkv' in self.source or 'flv' in self.source or 'mov' in self.source:
            cap = cv2.VideoCapture(self.source)
            self.total_frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
            cap.release()

        # 如果保存，则创建写入对象
        img_res, result, height, width = self.recognize_res(iter_model)
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        out = None  # 视频写出变量
        if self.save_res:
            out = cv2.VideoWriter(f'{self.save_res_path}/video_result_{video_id_count}.mp4', fourcc, 25,
                                  (width, height), True)  # 保存检测视频的路径

        # 开始死循环检测
        while True:
            try:
                # 暂停与开始
                if self.continue_dtc:
                    img_res, result, height, width = self.recognize_res(iter_model)
                    print('result:', result)
                    self.res_address(img_res, result, height, width, model, out)

                # 终止
                if self.stop_dtc:
                    if self.save_res:
                        if out:
                            out.release()
                            video_id_count += 1
                    self.source = None
                    self.yolo2main_status_msg.emit('检测终止')
                    self.release_capture()  # 这里是为了终止使用摄像头检测函数的线程，改了yolo源码
                    break

            # 检测截止（本地文件检测）
            except StopIteration:
                if self.save_res:
                    out.release()
                    video_id_count += 1
                    print('writing complete')
                self.yolo2main_status_msg.emit('检测完成')
                self.yolo2main_progress.emit(1000)
                # cv2.destroyAllWindows()  # 单目标追踪停止！
                self.source = None

                break
        try:
            out.release()
        except:
            pass

    # 进行识别——并返回所有结果
    def res_address(self, img_res, result, height, width, model, out):
            # 复制一份
            img_box = np.copy(img_res)   # 右边的图（会绘制标签！） img_res是原图-不会受影响
            img_trail = np.copy(img_res) # 左边的图

            # 如果没有识别的：
            if result.boxes.id is None:
                # 目标都是0
                self.sum_of_count = 0
                self.class_num = 0
                labels_write = "暂未识别到目标！"
            # 如果有识别的
            else:
                detections = sv.Detections.from_yolov8(result)
                detections.tracker_id = result.boxes.id.cpu().numpy().astype(int)

                # id 、位置、目标总数
                self.class_num = self.get_class_number(detections)  # 类别数
                id = detections.tracker_id  # id
                xyxy = detections.xyxy  # 位置
                self.sum_of_count = len(id)  # 目标总数

                img_trail = img_res  # 显示原图

                # 收集检测到的所有类别
                detected_classes = [model.model.names[i] for i in detections.class_id]


                # 画标签到图像上（并返回要写下的信息
                labels_write, img_box = self.creat_labels(detections, img_box , model)

                # 检测逻辑...
                if "good" not in detected_classes:  # 假设detected_classes包含了所有检测到的类别
                    self.non_good_detected.emit()  # 如果检测到的类别中不包含"good"，发出信号


            # 写入txt——存储labels里的信息
            if self.save_txt:
                with open(f'{self.save_txt_path}/result.txt', 'a') as f:
                    f.write('当前时刻屏幕信息:' +
                            str(labels_write) +
                            f'检测时间: {datetime.datetime.now().strftime("%Y-%m-%d-%H:%M:%S")}' +
                            f' 路段通过的目标总数: {self.sum_of_count}')
                    f.write('\n')

            # 预测视频写入本地
            if self.save_res:
                out.write(img_box)

            # 添加 折线图数据
            now = datetime.datetime.now()
            new_time = now.strftime("%Y-%m-%d %H:%M:%S")
            if new_time not in x_axis_time_graph:  # 防止同一秒写入
                x_axis_time_graph.append(new_time)
                y_axis_count_graph.append(self.sum_of_count)


            # 抠锚框里的图  （单目标追踪）
            if self.lock_id is not None:
                self.lock_id = int(self.lock_id)
                self.open_target_tracking(detections=detections, img_res=img_res)

            # 传递信号给主窗口
            self.emit_res(img_trail, img_box)

    # 识别结果处理
    def recognize_res(self, iter_model):
            # 检测 ---然后获取有用的数据
            result = next(iter_model)  # 这里是检测的核心，每次循环都会检测一帧图像,可以自行打印result看看里面有哪些key可以用
            img_res = result.orig_img  # 原图
            height, width, _ = img_res.shape

            return img_res, result, height, width


    def check_path(path):
        if not os.path.exists(path):
            os.mkdir(path)

    # 信号发送区
    def emit_res(self, img_trail, img_box):

        time.sleep(self.speed_thres/1000)  # 缓冲
        # 轨迹图像（左边）
        self.yolo2main_trail_img.emit(img_trail)
        # 标签图（右边）
        self.yolo2main_box_img.emit(img_box)
        # 总类别数量 、 总目标数
        self.yolo2main_class_num.emit(self.class_num)
        self.yolo2main_target_num.emit(self.sum_of_count)
        # 进度条
        if '0' in self.source or 'rtsp' in self.source:
            self.yolo2main_progress.emit(0)
        else:
            self.progress_value = int(self.count / self.total_frames * 1000)
            self.yolo2main_progress.emit(self.progress_value)
        # 计算FPS
        self.count += 1
        if self.count % 3 == 0 and self.count >= 3:  # 计算FPS
            self.yolo2main_fps.emit(str(int(3 / (time.time() - self.start_time))))
            self.start_time = time.time()

    # 加载模型
    def load_yolo_model(self):
        if self.used_model_name != self.new_model_name:
            self.setup_model(self.new_model_name)
            self.used_model_name = self.new_model_name
        return YOLO(self.new_model_name)
    # 画标签到图像上
    def creat_labels(self, detections, img_box, model):
        # 要画出来的信息
        # labels_draw = [
        #     f"ID: {tracker_id} {model.model.names[class_id]}"
        #     for _, _, confidence, class_id, tracker_id in detections
        # ]
        labels_draw = [
            f"ID: {tracker_id} CLS: {model.model.names[class_id]} CF: {confidence:0.2f}"
            for _,_,confidence,class_id,tracker_id in detections
        ]
        '''
        如果Torch装的是cuda版本的话：302行的代码需改成：
          labels_draw = [
            f"OBJECT-ID: {tracker_id} CLASS: {model.model.names[class_id]} CF: {confidence:0.2f}"
            for _,confidence,class_id,tracker_id in detections
        ]
        '''
        # 存储labels里的信息
        # labels_write = [
        #     f"目标ID: {tracker_id} 目标类别: {class_id} 置信度: {confidence:0.2f}"
        #     for _, _, confidence, class_id, tracker_id in detections
        # ]
        labels_write = [
            f"ID: {tracker_id} CLASS: {model.model.names[class_id]} CF: {confidence:0.2f}"
            for _, _,  confidence, class_id, tracker_id in detections
        ]
        '''
          如果Torch装的是cuda版本的话：314行的代码需改成：
          labels_write = [
            f"OBJECT-ID: {tracker_id} CLASS: {model.model.names[class_id]} CF: {confidence:0.2f}"
            for _,confidence,class_id,tracker_id in detections
        ]
        '''

        # 如果显示标签 （要有才可以画呀！）---否则就是原图
        if (self.show_labels == True) and (self.class_num != 0):
            img_box = self.box_annotator.annotate(scene=img_box, detections=detections, labels=labels_draw)

        return labels_write, img_box

    # 获取类别数
    def get_class_number(self, detections):
        class_num_arr = []
        for each in detections.class_id:
            if each not in class_num_arr:
                class_num_arr.append(each)
        return len(class_num_arr)

    # 释放摄像头
    def release_capture(self):
        LoadStreams.capture = 'release'  # 这里是为了终止使用摄像头检测函数的线程，改了yolo源码

class MainWindow(QMainWindow, Ui_MainWindow):
    main2yolo_begin_sgl = Signal()  # The main window sends an execution signal to the yolo instance
    def __init__(self, parent=None):
        super(MainWindow, self).__init__(parent)
        # basic interface
        self.setupUi(self)
        self.setAttribute(Qt.WA_TranslucentBackground)  # rounded transparent
        self.setWindowFlags(Qt.FramelessWindowHint)  # Set window flag: hide window borders
        UIFuncitons.uiDefinitions(self)
        # Show module shadows
        UIFuncitons.shadow_style(self, self.Class_QF, QColor(162,129,247))
        UIFuncitons.shadow_style(self, self.Target_QF, QColor(251, 157, 139))
        UIFuncitons.shadow_style(self, self.Fps_QF, QColor(170, 128, 213))
        UIFuncitons.shadow_style(self, self.Model_QF, QColor(64, 186, 193))



        # read model folder
        self.pt_list = os.listdir('./models')
        self.pt_list = [file for file in self.pt_list if file.endswith('.pt')]
        self.pt_list.sort(key=lambda x: os.path.getsize('./models/' + x))   # sort by file size
        self.model_box.clear()
        self.model_box.addItems(self.pt_list)
        self.Qtimer_ModelBox = QTimer(self)     # Timer: Monitor model file changes every 2 seconds
        self.Qtimer_ModelBox.timeout.connect(self.ModelBoxRefre)
        self.Qtimer_ModelBox.start(2000)

        # Yolo-v8 thread
        self.yolo_predict = YoloPredictor()                           # Create a Yolo instance
        self.select_model = self.model_box.currentText()                   # default model
        self.yolo_predict.new_model_name = "./models/%s" % self.select_model
        self.yolo_thread = QThread()                                  # Create yolo thread
        self.yolo_predict.yolo2main_trail_img.connect(lambda x: self.show_image(x, self.pre_video))
        self.yolo_predict.yolo2main_box_img.connect(lambda x: self.show_image(x, self.res_video))
        self.yolo_predict.yolo2main_status_msg.connect(lambda x: self.show_status(x))
        self.yolo_predict.yolo2main_fps.connect(lambda x: self.fps_label.setText(x))
        # self.yolo_predict.yolo2main_labels.connect(self.show_labels)
        self.yolo_predict.yolo2main_class_num.connect(lambda x:self.Class_num.setText(str(x)))
        self.yolo_predict.yolo2main_target_num.connect(lambda x:self.Target_num.setText(str(x)))
        self.yolo_predict.yolo2main_progress.connect(lambda x: self.progress_bar.setValue(x))
        self.main2yolo_begin_sgl.connect(self.yolo_predict.run)
        self.yolo_predict.moveToThread(self.yolo_thread)
        self.yolo_predict.non_good_detected.connect(self.show_non_good_warning)


        # Model parameters
        self.model_box.currentTextChanged.connect(self.change_model)
        self.iou_spinbox.valueChanged.connect(lambda x:self.change_val(x, 'iou_spinbox'))    # iou box
        self.iou_slider.valueChanged.connect(lambda x:self.change_val(x, 'iou_slider'))      # iou scroll bar
        self.conf_spinbox.valueChanged.connect(lambda x:self.change_val(x, 'conf_spinbox'))  # conf box
        self.conf_slider.valueChanged.connect(lambda x:self.change_val(x, 'conf_slider'))    # conf scroll bar
        self.speed_spinbox.valueChanged.connect(lambda x:self.change_val(x, 'speed_spinbox'))# speed box
        self.speed_slider.valueChanged.connect(lambda x:self.change_val(x, 'speed_slider'))  # speed scroll bar

        # Prompt window initialization
        self.Class_num.setText('--')
        self.Target_num.setText('--')
        self.fps_label.setText('--')
        self.Model_name.setText(self.select_model)

        # Select detection source
        self.src_file_button.clicked.connect(self.open_src_file)  # select local file
        self.src_cam_button.clicked.connect(self.chose_cam)#chose_cam
        # self.src_rtsp_button.clicked.connect(self.rtsp_seletction)#chose_rtsp
        self.src_web_button.clicked.connect(self.show_status("The function has not yet been implemented."))#chose_web
        # self.src_cam_button.clicked.connect(self.show_status("The function has not yet been implemented."))#chose_cam
        # self.src_rtsp_button.clicked.connect(self.show_status("The function has not yet been implemented."))#chose_rtsp

        # start testing button
        self.run_button.clicked.connect(self.run_or_continue)   # pause/start
        self.stop_button.clicked.connect(self.stop)             # termination

        # Other function buttons
        self.save_res_button.toggled.connect(self.is_save_res)  # save image option
        self.save_txt_button.toggled.connect(self.is_save_txt)  # Save label option
        self.ToggleBotton.clicked.connect(lambda: UIFuncitons.toggleMenu(self, True))   # left navigation button
        self.settings_button.clicked.connect(lambda: UIFuncitons.settingBox(self, True))   # top right settings button

        self.show_warning_again = True  # 新增状态变量

        # initialization
        self.load_config()

    # The main window displays the original image and detection results
    @staticmethod
    def show_image(img_src, label):
        try:
            # 检查图像的通道数，确定图像是否为彩色图像
            if len(img_src.shape) == 3:
                ih, iw, _ = img_src.shape
            if len(img_src.shape) == 2:
                ih, iw = img_src.shape

            # 根据标签窗口的大小调整图像的大小
            w = label.geometry().width()
            h = label.geometry().height()

            # 根据图像宽高比例进行缩放
            if iw / w > ih / h:
                scal = w / iw
                nw = w
                nh = int(scal * ih)
                img_src_ = cv2.resize(img_src, (nw, nh))

            else:
                scal = h / ih
                nw = int(scal * iw)
                nh = h
                img_src_ = cv2.resize(img_src, (nw, nh))

            # 将OpenCV图像从BGR格式转换为RGB格式，并创建QImage对象
            frame = cv2.cvtColor(img_src_, cv2.COLOR_BGR2RGB)
            img = QImage(frame.data, frame.shape[1], frame.shape[0], frame.shape[2] * frame.shape[1],
                         QImage.Format_RGB888)

            # 在标签窗口中显示图像
            label.setPixmap(QPixmap.fromImage(img))

        except Exception as e:
            print(repr(e))

    # Control start/pause
    def run_or_continue(self):
        # 检测是否有模型
        if self.yolo_predict.new_model_name == '' or self.yolo_predict.new_model_name == None:
            # Todo: 添加对话
            # DialogOver(parent=self, text="请检测模型", title="运行失败", flags="danger")
            self.run_button.setChecked(False)
            return
        # 检测输入源
        if self.yolo_predict.source == '' or self.yolo_predict.source == None:
            self.show_status('请在检测前选择输入源...')
            self.run_button.setChecked(False)
            # DialogOver(parent=self, text="请检测输入源", title="运行失败", flags="danger")
            return

        self.yolo_predict.stop_dtc = False # 线程开始

        # 开始
        if self.run_button.isChecked():

            # 图片预测
            file_extension = self.yolo_predict.source[-3:].lower()
            if file_extension == 'png' or file_extension == 'jpg':
                self.img_predict()
                return

            # 视频预测
            # DialogOver(parent=self, text="开始检测...", title="运行成功", flags="success")
            self.run_button.setChecked(True)


            # self.draw_thread.run_continue()  # 折线图开始

            # 不可再改变设置（config动态调整 关闭）
            self.save_txt_button.setEnabled(False)
            self.save_res_button.setEnabled(False)
            self.conf_slider.setEnabled(False)
            self.iou_slider.setEnabled(False)
            self.speed_slider.setEnabled(False)

            self.show_status('检测中...')
            if '0' in self.yolo_predict.source or 'rtsp' in self.yolo_predict.source:
                self.progress_bar.setFormat('实时视频流检测中...')
            if 'avi' in self.yolo_predict.source or 'mp4' in self.yolo_predict.source:
                self.progress_bar.setFormat("当前检测进度:%p%")
            self.yolo_predict.continue_dtc = True
            # 开始检测
            if not self.yolo_thread.isRunning():
                self.yolo_thread.start()
                self.main2yolo_begin_sgl.emit()
        # 暂停
        else:
            self.yolo_predict.continue_dtc = False
            self.show_status("暂停...")
            # DialogOver(parent=self, text="已暂停检测", title="运行暂停", flags="warning")
            self.run_button.setChecked(False)

    # bottom status bar information
    def show_status(self, msg):
        self.status_bar.setText(msg)
        if msg == 'Detection completed' or msg == '检测完成':
            self.save_res_button.setEnabled(True)
            self.save_txt_button.setEnabled(True)
            self.run_button.setChecked(False)
            self.progress_bar.setValue(0)
            if self.yolo_thread.isRunning():
                self.yolo_thread.quit()         # end process
        elif msg == 'Detection terminated!' or msg == '检测终止':
            self.save_res_button.setEnabled(True)
            self.save_txt_button.setEnabled(True)
            self.run_button.setChecked(False)
            self.progress_bar.setValue(0)
            if self.yolo_thread.isRunning():
                self.yolo_thread.quit()         # end process
            self.pre_video.clear()           # clear image display
            self.res_video.clear()
            self.Class_num.setText('--')
            self.Target_num.setText('--')
            self.fps_label.setText('--')

    # select local file
    def open_src_file(self):
        config_file = 'config/fold.json'
        config = json.load(open(config_file, 'r', encoding='utf-8'))
        open_fold = config['open_fold']
        if not os.path.exists(open_fold):
            open_fold = os.getcwd()
        name, _ = QFileDialog.getOpenFileName(self, 'Video/image', open_fold, "Pic File(*.mp4 *.mkv *.avi *.flv *.jpg *.png)")
        if name:
            self.yolo_predict.source = name
            self.show_status('Load File：{}'.format(os.path.basename(name)))
            config['open_fold'] = os.path.dirname(name)
            config_json = json.dumps(config, ensure_ascii=False, indent=2)
            with open(config_file, 'w', encoding='utf-8') as f:
                f.write(config_json)
            self.stop()

    # Select camera source----  have one bug
    def chose_cam(self):
        # try:
        # 关闭YOLO线程
        self.stop()
        # 获取本地摄像头数量
        _, cams = Camera().get_cam_num()
        popMenu = QMenu()
        popMenu.setFixedWidth(self.src_cam_button.width())
        popMenu.setStyleSheet('''
                                            QMenu {
                                            font-size: 20px;
                                            font-family: "Microsoft YaHei UI";
                                            font-weight: light;
                                            color:white;
                                            padding-left: 5px;
                                            padding-right: 5px;
                                            padding-top: 4px;
                                            padding-bottom: 4px;
                                            border-style: solid;
                                            border-width: 0px;
                                            border-color: rgba(255, 212, 255, 255);
                                            border-radius: 3px;
                                            background-color: rgba(16,155,226,50);
                                            }
                                            ''')

        for cam in cams:
            exec("action_%s = QAction('%s 号摄像头')" % (cam, cam))
            exec("popMenu.addAction(action_%s)" % cam)
        pos = QCursor.pos()
        action = popMenu.exec(pos)

        # 设置摄像头来源
        if action:
            str_temp = ''
            selected_stream_source = str_temp.join(filter(str.isdigit, action.text()))  # 获取摄像头号，去除非数字字符
            self.yolo_predict.source = selected_stream_source
            self.show_status(f'摄像头设备:{action.text()}')
            # DialogOver(parent=self, text=f"当前摄像头为: {action.text()}", title="摄像头选择成功", flags="success")

    def rtsp_seletction(self):
        self.rtsp_window = Window()
        self.rtsp_window.rtspEdit.setText(self.rtsp_ip)
        self.rtsp_window.show()
        # 如果点击则加载RTSP
        self.rtsp_window.rtspButton.clicked.connect(lambda: self.load_rtsp(self.rtsp_window.rtspEdit.text()))

        # 2、加载RTSP

    def load_rtsp(self, ip):
        ip = "rtsp://admin:admin888@192.168.1.2:555"
        MessageBox(self.close_button, title='提示', text='加载 rtsp...', time=1000, auto=True).exec()
        self.stop()  # 关闭YOLO线程

        self.yolo_predict.source = ip
        self.rtsp_ip = ip  # 写会ip
        self.rtsp_window.close()

        # 状态显示
        self.show_status(f'加载rtsp地址:{ip}')
        # DialogOver(parent=self, text=f"rtsp地址为: {ip}", title="RTSP加载成功", flags="success")
    # select network source
    def chose_rtsp(self):
        self.rtsp_window = Window()
        config_file = 'config/ip.json'
        if not os.path.exists(config_file):
            ip = "rtsp://admin:admin888@192.168.1.2:555"
            new_config = {"ip": ip}
            new_json = json.dumps(new_config, ensure_ascii=False, indent=2)
            with open(config_file, 'w', encoding='utf-8') as f:
                f.write(new_json)
        else:
            config = json.load(open(config_file, 'r', encoding='utf-8'))
            ip = config['ip']
        self.rtsp_window.rtspEdit.setText(ip)
        self.rtsp_window.show()
        self.rtsp_window.rtspButton.clicked.connect(lambda: self.load_rtsp(self.rtsp_window.rtspEdit.text()))

    # load network sources
    def load_rtsp(self, ip):
        try:
            self.stop()
            MessageBox(
                self.close_button, title='提示', text='加载 rtsp...', time=1000, auto=True).exec()
            self.yolo_predict.source = ip
            new_config = {"ip": ip}
            new_json = json.dumps(new_config, ensure_ascii=False, indent=2)
            with open('config/ip.json', 'w', encoding='utf-8') as f:
                f.write(new_json)
            self.show_status('Loading rtsp：{}'.format(ip))
            self.rtsp_window.close()
        except Exception as e:
            self.show_status('%s' % e)

    # Save test result button--picture/video
    def is_save_res(self):
        if self.save_res_button.checkState() == Qt.CheckState.Unchecked:
            self.show_status('NOTE: Run image results are not saved.')
            self.yolo_predict.save_res = False
        elif self.save_res_button.checkState() == Qt.CheckState.Checked:
            self.show_status('NOTE: Run image results will be saved.')
            self.yolo_predict.save_res = True

    # Save test result button -- label (txt)
    def is_save_txt(self):
        if self.save_txt_button.checkState() == Qt.CheckState.Unchecked:
            self.show_status('NOTE: Labels results are not saved.')
            self.yolo_predict.save_txt = False
        elif self.save_txt_button.checkState() == Qt.CheckState.Checked:
            self.show_status('NOTE: Labels results will be saved.')
            self.yolo_predict.save_txt = True

    # Configuration initialization  ~~~wait to change~~~
    def load_config(self):
        config_file = 'config/setting.json'
        if not os.path.exists(config_file):
            iou = 0.26
            conf = 0.33
            rate = 10
            save_res = 0
            save_txt = 0
            new_config = {"iou": iou,
                          "conf": conf,
                          "rate": rate,
                          "save_res": save_res,
                          "save_txt": save_txt
                          }
            new_json = json.dumps(new_config, ensure_ascii=False, indent=2)
            with open(config_file, 'w', encoding='utf-8') as f:
                f.write(new_json)
        else:
            config = json.load(open(config_file, 'r', encoding='utf-8'))
            if len(config) != 5:
                iou = 0.26
                conf = 0.33
                rate = 10
                save_res = 0
                save_txt = 0
            else:
                iou = config['iou']
                conf = config['conf']
                rate = config['rate']
                save_res = config['save_res']
                save_txt = config['save_txt']
        self.save_res_button.setCheckState(Qt.CheckState(save_res))
        self.yolo_predict.save_res = (False if save_res==0 else True )
        self.save_txt_button.setCheckState(Qt.CheckState(save_txt))
        self.yolo_predict.save_txt = (False if save_txt==0 else True )
        self.run_button.setChecked(False)
        self.show_status("Welcome~")

    # Terminate button and associated state
    def stop(self):
        if self.yolo_thread.isRunning():
            self.yolo_thread.quit()         # end thread
        self.yolo_predict.stop_dtc = True
        self.run_button.setChecked(False)    # start key recovery
        self.save_res_button.setEnabled(True)   # Ability to use the save button
        self.save_txt_button.setEnabled(True)   # Ability to use the save button
        self.pre_video.clear()           # clear image display
        self.res_video.clear()           # clear image display
        self.progress_bar.setValue(0)
        self.Class_num.setText('--')
        self.Target_num.setText('--')
        self.fps_label.setText('--')

    # Change detection parameters
    def change_val(self, x, flag):
        if flag == 'iou_spinbox':
            self.iou_slider.setValue(int(x*100))    # The box value changes, changing the slider
        elif flag == 'iou_slider':
            self.iou_spinbox.setValue(x/100)        # The slider value changes, changing the box
            self.show_status('IOU Threshold: %s' % str(x/100))
            self.yolo_predict.iou_thres = x/100
        elif flag == 'conf_spinbox':
            self.conf_slider.setValue(int(x*100))
        elif flag == 'conf_slider':
            self.conf_spinbox.setValue(x/100)
            self.show_status('Conf Threshold: %s' % str(x/100))
            self.yolo_predict.conf_thres = x/100
        elif flag == 'speed_spinbox':
            self.speed_slider.setValue(x)
        elif flag == 'speed_slider':
            self.speed_spinbox.setValue(x)
            self.show_status('Delay: %s ms' % str(x))
            self.yolo_predict.speed_thres = x  # ms

    # change model
    def change_model(self,x):
        self.select_model = self.model_box.currentText()
        self.yolo_predict.new_model_name = "./models/%s" % self.select_model
        self.show_status('Change Model：%s' % self.select_model)
        self.Model_name.setText(self.select_model)

    # label result
    # def show_labels(self, labels_dic):
    #     try:
    #         self.result_label.clear()
    #         labels_dic = sorted(labels_dic.items(), key=lambda x: x[1], reverse=True)
    #         labels_dic = [i for i in labels_dic if i[1]>0]
    #         result = [' '+str(i[0]) + '：' + str(i[1]) for i in labels_dic]
    #         self.result_label.addItems(result)
    #     except Exception as e:
    #         self.show_status(e)

    # Cycle monitoring model file changes
    def ModelBoxRefre(self):
        pt_list = os.listdir('./models')
        pt_list = [file for file in pt_list if file.endswith('.pt')]
        pt_list.sort(key=lambda x: os.path.getsize('./models/' + x))
        # It must be sorted before comparing, otherwise the list will be refreshed all the time
        if pt_list != self.pt_list:
            self.pt_list = pt_list
            self.model_box.clear()
            self.model_box.addItems(self.pt_list)

    # Get the mouse position (used to hold down the title bar and drag the window)
    def mousePressEvent(self, event):
        p = event.globalPosition()
        globalPos = p.toPoint()
        self.dragPos = globalPos

    # Optimize the adjustment when dragging the bottom and right edges of the window size
    def resizeEvent(self, event):
        # Update Size Grips
        UIFuncitons.resize_grips(self)

    # Exit Exit thread, save settings
    def closeEvent(self, event):
        config_file = 'config/setting.json'
        config = dict()
        config['iou'] = self.iou_spinbox.value()
        config['conf'] = self.conf_spinbox.value()
        config['rate'] = self.speed_spinbox.value()
        config['save_res'] = (0 if self.save_res_button.checkState()==Qt.Unchecked else 2)
        config['save_txt'] = (0 if self.save_txt_button.checkState()==Qt.Unchecked else 2)
        config_json = json.dumps(config, ensure_ascii=False, indent=2)
        with open(config_file, 'w', encoding='utf-8') as f:
            f.write(config_json)
        # Exit the process before closing
        if self.yolo_thread.isRunning():
            self.yolo_predict.stop_dtc = True
            self.yolo_thread.quit()
            MessageBox(
                self.close_button, title='Note', text='Exiting, please wait...', time=3000, auto=True).exec()
            sys.exit(0)
        else:
            sys.exit(0)

    def img_predict(self):

        # if check_url(self.yolo_predict.source):
        #     DialogOver(parent=self, text="目标路径含有中文！", title="程序取消", flags="danger")
        #     return

        self.run_button.setChecked(False)  # 按钮
        # 读取照片
        image = cv2.imread(self.yolo_predict.source)
        org_img = image.copy()
        # 加载模型
        model = self.yolo_predict.load_yolo_model()
        # 获取数据源
        iter_model = iter(model.track(source=image, show=False))
        result = next(iter_model)  # 这里是检测的核心，
        # 如果没有目标
        if result.boxes.id is None:
            # DialogOver(parent=self, text="该图片中没有要检测的目标哟！", title="运行完成", flags="warning")
            self.show_image(image, self.pre_video)
            self.show_image(image, self.res_video)
            self.yolo_predict.source = ''
            return

        # 如果有目标
        detections = sv.Detections.from_yolov8(result)
        detections.tracker_id = result.boxes.id.cpu().numpy().astype(int)
        # 画标签
        labels_write, img_box = self.yolo_predict.creat_labels(detections, image, model)

        # 显示信息 —— 类别数 & 总数
        self.Class_num.setText(str(self.yolo_predict.get_class_number(detections)))
        self.Target_num.setText(str(len(detections.tracker_id)))
        # 显示图片
        self.show_image(org_img, self.pre_video)  # left
        self.show_image(img_box, self.res_video)  # right
        self.yolo_predict.source = ''
        # DialogOver(parent=self, text="图片检测完成", title="运行成功", flags="success")

        # 保存图片
        if self.yolo_predict.save_res:
            # check_path(self.config.save_res_path) # 检查保存路径
            # 存在同名文件，自增 self.image_id 直至文件不存在
            while os.path.exists(f"{self.config.save_res_path}/image_result_{self.image_id}.jpg"):
                self.image_id += 1
            # 将 BGR 格式的 frame 转换为 RGB 格式
            rgb_frame = cv2.cvtColor(img_box, cv2.COLOR_BGR2RGB)
            # 把 rgb_frame 转换为 numpy格式 就行了
            numpy_frame = np.array(rgb_frame)
            Image.fromarray(numpy_frame).save(f"./{self.config.save_res_path}/image_result_{self.image_id}.jpg")

        # 存储labels里的信息
        if self.yolo_predict.save_txt:
            # check_path(self.config.save_txt_path) # 检查保存路径
            # 存在同名文件，自增 self.txt_id 直至文件不存在
            while os.path.exists(f"{self.config.save_txt_path}/result_{self.txt_id}.jpg"):
                self.txt_id += 1

            with open(f'{self.config.save_txt_path}/result_{self.txt_id}.txt', 'a') as f:
                f.write('当前时刻屏幕信息:' +
                        str(labels_write) +
                        f'检测时间: {datetime.datetime.now().strftime("%Y-%m-%d-%H:%M:%S")}')
                f.write('\n')
        return

    def show_non_good_warning(self,):
        if self.show_warning_again:
            self.yolo_predict.continue_dtc = False
            self.show_status("暂停...")
            self.run_button.setChecked(False)

            # 使用自定义的对话框来显示警告
            dialog = WarningDialog(self)
            if dialog.exec_():  # 如果用户选择了"不再提示"
                self.show_warning_again = False  # 更新状态，不再显示警告


class WarningDialog(QDialog):
    def __init__(self, parent=None):
        super(WarningDialog, self).__init__(parent)
        self.setWindowTitle("检测结果")
        self.setMinimumSize(300, 150)  # 设置对话框的最小尺寸
        self.setStyleSheet("QDialog { background-color: #f0f0f0; }")  # 对话框背景色

        self.layout = QVBoxLayout(self)

        self.label = QLabel("检测到异常类别！", self)
        # 设置标签字体大小，文本颜色为红色
        self.label.setStyleSheet("""
            QLabel { 
                font-size: 16px; 
                color: red;
            }
        """)
        # 设置标签文本居中
        self.label.setAlignment(Qt.AlignCenter)
        self.layout.addWidget(self.label)

        self.checkbox = QCheckBox("不再提示此警告", self)
        self.checkbox.setStyleSheet("QCheckBox { font-size: 14px; }")  # 复选框字体大小
        self.layout.addWidget(self.checkbox)

        self.button = QPushButton("确定", self)
        self.button.setStyleSheet(
            """
            QPushButton {
                font-size: 14px;
                color: white;
                background-color: #007bff;
                border-radius: 10px;
                padding: 10px;
            }
            QPushButton:hover {
                background-color: #0056b3;
            }
            """
        )  # 按钮样式
        self.button.clicked.connect(self.accept)
        self.layout.addWidget(self.button)

        self.checkbox_state = False
        self.checkbox.stateChanged.connect(self.update_checkbox_state)

    def update_checkbox_state(self, state):
        self.checkbox_state = state == Qt.Checked

    def exec_(self):
        super(WarningDialog, self).exec_()
        return self.checkbox.isChecked()



if __name__ == "__main__":
    app = QApplication(sys.argv)
    Home = MainWindow()
    Home.show()
    sys.exit(app.exec())
