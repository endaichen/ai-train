"""
单视频跟踪计数脚本 — 画线后自动跟踪目标并统计越线数量
"""
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

from ultralytics import YOLO
import cv2
import numpy as np
import yaml
from PIL import Image, ImageDraw, ImageFont

# ======================== 常量配置 ========================
# 时序稳定性参数(可根据实际情况微调)
DET_CONF = 0.20       # 检测置信度阈值(漏检多→降低)
NMS_IOU = 0.5         # NMS的IoU阈值(多余框多→增大)
IOU_DEDUP = 0.5       # 同类别框去重IoU(分裂框多→降低)
CONF_WEAK = 0.35      # 弱检测判定线(闪框多→提高,漏框多→降低)
MIN_TRACK_AGE = 3     # 轨迹最短存活帧数(闪框多→增大)

# 路径配置
MODEL_PATH = ""       # 模型权重路径
VIDEO_PATH = ""       # 视频文件路径
YAML_PATH = ""        # 类别配置文件路径
OUTPUT_DIR = ""       # 输出文件夹路径
# ======================================================

# ======================== 中文字体(模块级缓存) ========================
_cached_font = None

def _get_font(size=24):
    """获取中文字体，仅首次加载后缓存"""
    global _cached_font
    if _cached_font is not None:
        return _cached_font
    for name in ("simhei.ttf", "msyh.ttc"):
        try:
            _cached_font = ImageFont.truetype(name, size)
            return _cached_font
        except OSError:
            continue
    _cached_font = ImageFont.load_default()
    return _cached_font


# ======================== 画线交互 ========================
class LineDrawer:
    """在图像上交互式画线，用于设定计数线"""

    def __init__(self, img):
        self.img = img.copy()
        self.drawing = False
        self.start_point = None
        self.end_point = None
        self.line = None

    def mouse_callback(self, event, x, y, flags, param):
        """鼠标回调：左键按下→移动→抬起完成画线"""
        if event == cv2.EVENT_LBUTTONDOWN:
            self.drawing = True
            self.start_point = self.end_point = (x, y)
        elif event == cv2.EVENT_MOUSEMOVE and self.drawing:
            self.end_point = (x, y)
        elif event == cv2.EVENT_LBUTTONUP:
            self.drawing = False
            self.end_point = (x, y)
            self.line = (self.start_point, self.end_point)

    def draw(self):
        """绘制当前线段和中文提示信息，返回BGR图像"""
        temp = self.img.copy()
        if self.start_point and self.end_point:
            cv2.line(temp, self.start_point, self.end_point, (0, 255, 0), 2)

        # 用PIL绘制中文文字
        temp_pil = Image.fromarray(cv2.cvtColor(temp, cv2.COLOR_BGR2RGB))
        draw = ImageDraw.Draw(temp_pil)
        draw.text((10, 5), "画线后按空格确认，直接按空格用中点线，按回车跳过",
                  font=_get_font(), fill=(0, 255, 0))
        return cv2.cvtColor(np.array(temp_pil), cv2.COLOR_RGB2BGR)


def get_line_from_user(img):
    """弹出窗口让用户在图像上画线，返回 (start, end) 或 None(跳过)"""
    drawer = LineDrawer(img)
    cv2.namedWindow('Draw Line')
    cv2.setMouseCallback('Draw Line', drawer.mouse_callback)

    h, w = img.shape[:2]
    default_line = ((0, h // 2), (w, h // 2))  # 默认水平中线

    while True:
        cv2.imshow('Draw Line', drawer.draw())
        key = cv2.waitKey(1) & 0xFF
        if key == ord(' '):  # 空格确认
            if drawer.line is None:
                drawer.line = default_line
                print(f"未画线，使用默认中点线: {default_line}")
            break
        elif key == 13:  # 回车跳过
            print("按回车键跳过")
            drawer.line = None
            break

    cv2.destroyWindow('Draw Line')
    return drawer.line


# ======================== 几何与去重工具 ========================
def point_to_line_distance(point, line_start, line_end):
    """点到有向直线的带符号距离(正值在一侧，负值在另一侧)"""
    line_vec = np.array(line_end) - np.array(line_start)
    point_vec = np.array(point) - np.array(line_start)
    return np.cross(line_vec, point_vec) / np.linalg.norm(line_vec)


def compute_iou(box1, box2):
    """计算两个bbox的IoU，用于框去重"""
    x1, y1 = max(box1[0], box2[0]), max(box1[1], box2[1])
    x2, y2 = min(box1[2], box2[2]), min(box1[3], box2[3])
    if x2 <= x1 or y2 <= y1:
        return 0.0
    inter = (x2 - x1) * (y2 - y1)
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    return inter / (area1 + area2 - inter + 1e-6)


def deduplicate_boxes(boxes_with_info, iou_thresh=0.5):
    """
    同类别重叠框去重：按置信度降序排列，高置信度优先保留。
    boxes_with_info: [(x1,y1,x2,y2, cls_id, conf, track_id), ...]
    """
    if len(boxes_with_info) <= 1:
        return boxes_with_info
    sorted_boxes = sorted(boxes_with_info, key=lambda x: x[5], reverse=True)
    keep = []
    for box in sorted_boxes:
        if not any(box[4] == k[4] and compute_iou(box[:4], k[:4]) > iou_thresh for k in keep):
            keep.append(box)
    return keep


# ======================== 配置工具 ========================
def load_class_names_from_yaml(yaml_path):
    """从yaml配置文件中加载类别名称"""
    if not os.path.exists(yaml_path):
        print(f"YAML配置文件不存在: {yaml_path}")
        return None
    with open(yaml_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    if 'names' not in config:
        print("YAML配置文件中未找到 'names' 字段")
        return None
    print(f"从配置文件加载到类别: {config['names']}")
    return config['names']


def find_target_class(yaml_classes, model_classes, keyword="hao"):
    """
    在yaml类别中查找包含keyword的标签，并匹配模型中的类别ID。
    返回 (target_label, hao_class_id) 或 (None, None)
    """
    if isinstance(yaml_classes, list):
        items = enumerate(yaml_classes)
    elif isinstance(yaml_classes, dict):
        items = yaml_classes.items()
    else:
        return None, None

    target_label = None
    for _, name in items:
        if isinstance(name, str) and keyword in name.lower():
            target_label = name
            break

    if target_label is None:
        return None, None

    for cls_id, cls_name in model_classes.items():
        if cls_name == target_label:
            return target_label, cls_id
    return target_label, None


# ======================== 主流程 ========================
def main():
    # 检查文件是否存在
    for path, desc in [(MODEL_PATH, "模型文件"), (VIDEO_PATH, "视频文件"), (YAML_PATH, "YAML配置文件")]:
        if not os.path.exists(path):
            print(f"{desc}不存在: {path}")
            return

    # 加载模型
    print("正在加载模型...")
    model = YOLO(MODEL_PATH)

    import torch
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"使用设备: {device.upper()}")
    model.to(device)

    # 打开视频并获取第一帧
    cap = cv2.VideoCapture(VIDEO_PATH)
    if not cap.isOpened():
        print(f"无法打开视频: {VIDEO_PATH}")
        return

    ret, first_frame = cap.read()
    if not ret:
        print("无法读取视频第一帧")
        return

    # 让用户画起始线
    print("请在视频中画一条起始线，画完按空格确认")
    line = get_line_from_user(first_frame)
    if line is None:
        print("未画线，退出")
        return
    line_start, line_end = line
    print(f"起始线: {line_start} -> {line_end}")

    # 重置视频到开头并获取信息
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # 创建输出
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    video_name_no_ext = os.path.splitext(os.path.basename(VIDEO_PATH))[0]
    output_path = os.path.join(OUTPUT_DIR, f"{video_name_no_ext}_AI分析.avi")
    out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'XVID'), fps, (width, height))

    # 加载类别并查找目标
    yaml_classes = load_class_names_from_yaml(YAML_PATH)
    model_classes = model.names
    print("模型中的类别:", model_classes)

    target_label, hao_class_id = find_target_class(yaml_classes, model_classes)
    if target_label is None:
        print("在YAML配置文件中未找到包含 'hao' 的标签")
        return
    if hao_class_id is None:
        print(f"模型中未找到 '{target_label}' 类别")
        return
    print(f"自动检测到目标标签: {target_label} (class_id={hao_class_id})")

    print(f"开始处理视频... 共 {total_frames} 帧")
    print("按任意键继续开始推理...")
    cv2.waitKey(0)

    # 计数与跟踪状态
    plus_count = minus_count = 0
    prev_positions = {}
    track_age = {}
    prev_track_ids = set()
    use_half = (device == "cuda")

    frame_idx = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # 跟踪推理
        track_kwargs = dict(persist=True, verbose=False, half=use_half, device=device,
                            conf=DET_CONF, iou=NMS_IOU)
        try:
            results = model.track(frame, **track_kwargs)
        except RuntimeError as e:
            print(f"CUDA错误(第{frame_idx}帧): {e}, 尝试清理GPU缓存继续")
            import torch
            torch.cuda.empty_cache()
            results = model.track(frame, **track_kwargs)

        current_positions = {}
        current_track_ids = set()

        if results[0].boxes is not None:
            boxes = results[0].boxes
            has_track_ids = boxes.id is not None

            # 步骤1: 收集目标类别的检测框
            raw_boxes = []
            for i, (box_xyxy, cls_id, conf) in enumerate(zip(boxes.xyxy, boxes.cls, boxes.conf)):
                if int(cls_id) != hao_class_id:
                    continue
                x1, y1, x2, y2 = (float(v) for v in box_xyxy)
                tid = int(boxes.id[i]) if has_track_ids else -1
                raw_boxes.append((x1, y1, x2, y2, int(cls_id), float(conf), tid))

            # 步骤2: 同类别重叠框去重
            merged_boxes = deduplicate_boxes(raw_boxes, IOU_DEDUP)

            # 步骤3: 时序过滤 + 绘制
            for (x1, y1, x2, y2, cls_id, conf, tid) in merged_boxes:
                # 闪框抑制: 新出现 + 低置信度 → 丢弃
                if tid not in prev_track_ids and conf < CONF_WEAK:
                    continue

                track_age[tid] = track_age.get(tid, 0) + 1
                center_x, center_y = int((x1 + x2) / 2), int((y1 + y2) / 2)

                # 绘制检测框和标签
                if has_track_ids and tid >= 0:
                    current_track_ids.add(tid)
                    current_positions[tid] = (center_x, center_y)
                    id_text = f"ID:{tid} {conf:.2f} a{track_age[tid]}"
                else:
                    id_text = f"{target_label} {conf:.2f}"

                color = (0, 255, 255) if conf < CONF_WEAK else (255, 0, 0)
                cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
                cv2.circle(frame, (center_x, center_y), 5, (0, 255, 255), -1)
                cv2.putText(frame, id_text, (int(x1), int(y1) - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

            # 清理已消失的轨迹年龄记录
            track_age = {tid: age for tid, age in track_age.items() if tid in current_track_ids}

        # 步骤4: 越线计数(仅对稳定轨迹)
        for track_id, curr_pos in current_positions.items():
            if track_age.get(track_id, 0) < MIN_TRACK_AGE:
                continue
            if track_id in prev_positions:
                prev_dist = point_to_line_distance(prev_positions[track_id], line_start, line_end)
                curr_dist = point_to_line_distance(curr_pos, line_start, line_end)
                # 异号表示穿越了计数线
                if prev_dist * curr_dist < -0.0001:
                    if curr_pos[1] > prev_positions[track_id][1]:
                        plus_count += 1
                        print(f"ID {track_id}: plus +1 (从上往下)")
                    else:
                        minus_count += 1
                        print(f"ID {track_id}: minus +1 (从下往上)")

        prev_positions = current_positions.copy()
        prev_track_ids = current_track_ids.copy()

        # 绘制计数线和统计信息
        report_count = plus_count - minus_count
        cv2.line(frame, line_start, line_end, (0, 255, 0), 2)
        cv2.putText(frame, f"Plus: {plus_count}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 3)
        cv2.putText(frame, f"Minus: {minus_count}", (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)
        cv2.putText(frame, f"Report: {report_count}", (20, 120), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 3)

        if frame_idx % 30 == 0:
            print(f"处理进度: {frame_idx}/{total_frames} ({frame_idx * 100 / total_frames:.1f}%)")

        out.write(frame)
        cv2.imshow('Tracking', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        frame_idx += 1

    # 释放资源
    cap.release()
    out.release()
    cv2.destroyAllWindows()

    # 重命名输出视频(追加计数结果)
    final_path = os.path.join(OUTPUT_DIR, f"{video_name_no_ext}_AI分析_{report_count}.avi")
    if os.path.exists(output_path):
        if os.path.exists(final_path):
            try:
                os.remove(final_path)
            except Exception as e:
                print(f"删除旧文件失败: {e}")
        os.rename(output_path, final_path)
        output_path = final_path

    print(f"\n处理完成！Plus: {plus_count}, Minus: {minus_count}, Report: {report_count}")
    print(f"输出视频已保存至: {output_path}")


if __name__ == '__main__':
    main()
