import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

from ultralytics import YOLO
import cv2
import numpy as np
import yaml
from PIL import Image, ImageDraw, ImageFont
import tkinter as tk
from tkinter import filedialog, messagebox

# ======================== 常量配置 ========================
VIDEO_EXTENSIONS = {'.mp4', '.avi', '.mov', '.mkv', '.flv', '.wmv', '.webm'}

# 时序稳定性参数(可根据实际情况微调)
DET_CONF = 0.15       # 检测置信度阈值(漏检多→降低)
NMS_IOU = 0.5         # NMS的IoU阈值(多余框多→增大)
IOU_DEDUP = 0.5       # 同类别框去重IoU(分裂框多→降低)
CONF_WEAK = 0.3       # 弱检测判定线(闪框多→提高,漏框多→降低)
MIN_TRACK_AGE = 3     # 轨迹最短存活帧数(闪框多→增大)

# 路径配置
MODEL_PATH = "E:/模型训练/cxd/runs/detect/train-2026-06-04-16/weights/best.pt"
YAML_PATH = "E:/模型训练/cxd/xunlianji/classes.yaml"
OUTPUT_BASE_DIR = "E:/模型训练/cxd/track_output_1"

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

    def draw(self, video_name=""):
        """绘制当前线段和中文提示信息，返回BGR图像"""
        temp = self.img.copy()
        if self.start_point and self.end_point:
            cv2.line(temp, self.start_point, self.end_point, (0, 255, 0), 2)

        # 用PIL绘制中文文字
        temp_pil = Image.fromarray(cv2.cvtColor(temp, cv2.COLOR_BGR2RGB))
        draw = ImageDraw.Draw(temp_pil)
        font = _get_font()

        y_offset = 5
        if video_name:
            draw.text((10, y_offset), f"当前视频: {video_name}", font=font, fill=(255, 255, 0))
            y_offset = 35
        draw.text((10, y_offset), "画线后按空格确认，直接按空格用中点线，按回车跳过", font=font, fill=(0, 255, 0))

        return cv2.cvtColor(np.array(temp_pil), cv2.COLOR_RGB2BGR)


def get_line_from_user(img, video_name=""):
    """弹出窗口让用户在图像上画线，返回 (start, end) 或 None(跳过)"""
    drawer = LineDrawer(img)
    cv2.namedWindow('Draw Line')
    cv2.setMouseCallback('Draw Line', drawer.mouse_callback)

    h, w = img.shape[:2]
    default_line = ((0, h // 2), (w, h // 2))

    while True:
        cv2.imshow('Draw Line', drawer.draw(video_name))
        key = cv2.waitKey(1) & 0xFF
        if key == ord(' '):
            if drawer.line is None:
                drawer.line = default_line
                print(f"未画线，使用默认中点线: {default_line}")
            break
        elif key == 13:
            print("按回车键跳过当前视频")
            drawer.line = None
            break

    cv2.destroyWindow('Draw Line')
    return drawer.line


# ======================== 几何与去重工具 ========================
def point_to_line_distance(point, line_start, line_end):
    """点到有向直线的带符号距离"""
    line_vec = np.array(line_end) - np.array(line_start)
    point_vec = np.array(point) - np.array(line_start)
    return np.cross(line_vec, point_vec) / np.linalg.norm(line_vec)


def compute_iou(box1, box2):
    """计算两个bbox的IoU"""
    x1, y1 = max(box1[0], box2[0]), max(box1[1], box2[1])
    x2, y2 = min(box1[2], box2[2]), min(box1[3], box2[3])
    if x2 <= x1 or y2 <= y1:
        return 0.0
    inter = (x2 - x1) * (y2 - y1)
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    return inter / (area1 + area2 - inter + 1e-6)


def deduplicate_boxes(boxes_with_info, iou_thresh=0.5):
    """同类别重叠框去重"""
    if len(boxes_with_info) <= 1:
        return boxes_with_info
    sorted_boxes = sorted(boxes_with_info, key=lambda x: x[5], reverse=True)
    keep = []
    for box in sorted_boxes:
        if not any(box[4] == k[4] and compute_iou(box[:4], k[:4]) > iou_thresh for k in keep):
            keep.append(box)
    return keep


# ======================== 配置与文件工具 ========================
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


def select_folder():
    """使用tkinter选择文件夹"""
    root = tk.Tk()
    root.withdraw()
    folder_path = filedialog.askdirectory(title="选择包含待处理视频的文件夹")
    root.destroy()
    return folder_path


def get_video_files(folder_path):
    """获取文件夹中所有视频文件"""
    return sorted(
        os.path.join(folder_path, f)
        for f in os.listdir(folder_path)
        if os.path.splitext(f)[1].lower() in VIDEO_EXTENSIONS
    )


def select_video_count(max_count):
    """弹出输入框让用户选择要处理的视频数量"""
    root = tk.Tk()
    root.title("选择处理数量")
    root.geometry(f"400x200+{(root.winfo_screenwidth()-400)//2}+{(root.winfo_screenheight()-200)//2}")

    result = {"count": None}

    def validate_input():
        text = entry.get().strip()
        if not text:
            messagebox.showwarning("警告", "请输入数字！")
            return
        try:
            count = int(text)
        except ValueError:
            messagebox.showwarning("警告", "请输入有效的正整数！")
            return
        if count == 0:
            result["count"] = 0
            root.destroy()
        elif count < 0:
            messagebox.showwarning("警告", "请输入正整数！")
        elif count > max_count:
            messagebox.showwarning("警告", f"不能超过文件夹中的视频数量：{max_count}")
        else:
            result["count"] = count
            root.destroy()

    tk.Label(root, text=f"文件夹中共有 {max_count} 个视频文件", font=("Arial", 12)).pack(pady=15)
    tk.Label(root, text="请输入要处理的视频数量：", font=("Arial", 10)).pack(pady=5)
    entry = tk.Entry(root, font=("Arial", 12), width=20)
    entry.pack(pady=10)
    entry.focus_set()
    tk.Button(root, text="确认", command=validate_input, font=("Arial", 11), width=15).pack(pady=10)
    root.bind("<Return>", lambda e: validate_input())
    root.mainloop()
    return result["count"]


def find_target_class(yaml_classes, model_classes, keyword="hao"):
    """在yaml类别中查找包含keyword的标签"""
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


# ======================== 视频处理核心 ========================
def process_single_video(model, video_path, line, hao_class_id, yaml_classes, output_dir, device="cpu"):
    """处理单个视频"""
    video_name = os.path.basename(video_path)
    video_name_no_ext = os.path.splitext(video_name)[0]
    print(f"\n开始处理: {video_name}")

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"无法打开视频: {video_path}")
        return None

    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    output_path = os.path.join(output_dir, f"{video_name_no_ext}_AI分析.avi")
    out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'XVID'), fps, (width, height))

    plus_count = minus_count = 0
    prev_positions = {}
    track_age = {}
    prev_track_ids = set()
    line_start, line_end = line
    frame_idx = 0
    use_half = (device == "cuda")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

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

            raw_boxes = []
            for i, (box_xyxy, cls_id, conf) in enumerate(zip(boxes.xyxy, boxes.cls, boxes.conf)):
                if int(cls_id) != hao_class_id:
                    continue
                x1, y1, x2, y2 = (float(v) for v in box_xyxy)
                tid = int(boxes.id[i]) if has_track_ids else -1
                raw_boxes.append((x1, y1, x2, y2, int(cls_id), float(conf), tid))

            merged_boxes = deduplicate_boxes(raw_boxes, IOU_DEDUP)

            for (x1, y1, x2, y2, cls_id, conf, tid) in merged_boxes:
                if tid not in prev_track_ids and conf < CONF_WEAK:
                    continue

                track_age[tid] = track_age.get(tid, 0) + 1
                center_x, center_y = int((x1 + x2) / 2), int((y1 + y2) / 2)

                if has_track_ids and tid >= 0:
                    current_track_ids.add(tid)
                    current_positions[tid] = (center_x, center_y)
                    id_text = f"ID:{tid} {conf:.2f} a{track_age[tid]}"
                else:
                    label = yaml_classes[int(cls_id)] if yaml_classes else int(cls_id)
                    id_text = f"{label} {conf:.2f}"

                color = (0, 255, 255) if conf < CONF_WEAK else (255, 0, 0)
                cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
                cv2.circle(frame, (center_x, center_y), 5, (0, 255, 255), -1)
                cv2.putText(frame, id_text, (int(x1), int(y1) - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

            track_age = {tid: age for tid, age in track_age.items() if tid in current_track_ids}

        for track_id, curr_pos in current_positions.items():
            if track_age.get(track_id, 0) < MIN_TRACK_AGE:
                continue
            if track_id in prev_positions:
                prev_dist = point_to_line_distance(prev_positions[track_id], line_start, line_end)
                curr_dist = point_to_line_distance(curr_pos, line_start, line_end)
                if prev_dist * curr_dist < -0.0001:
                    if curr_pos[1] > prev_positions[track_id][1]:
                        plus_count += 1
                        print(f"ID {track_id}: plus +1")
                    else:
                        minus_count += 1
                        print(f"ID {track_id}: minus +1")

        prev_positions = current_positions.copy()
        prev_track_ids = current_track_ids.copy()

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

    cap.release()
    out.release()
    cv2.destroyAllWindows()

    final_path = os.path.join(output_dir, f"{video_name_no_ext}_AI分析_{report_count}.avi")
    if os.path.exists(output_path):
        if os.path.exists(final_path):
            try:
                os.remove(final_path)
            except Exception as e:
                print(f"删除旧文件失败: {e}")
        os.rename(output_path, final_path)

    print(f"\n{video_name} 处理完成！Plus: {plus_count}, Minus: {minus_count}, Report: {report_count}")
    return final_path, report_count


# ======================== 主流程 ========================
def main():
    for path, desc in [(MODEL_PATH, "模型文件"), (YAML_PATH, "YAML配置文件")]:
        if not os.path.exists(path):
            msg = f"{desc}不存在:\n{path}"
            print(msg)
            messagebox.showerror("错误", msg)
            return

    print("请选择包含待处理视频的文件夹...")
    video_folder = select_folder()
    if not video_folder:
        print("未选择文件夹，退出")
        return

    video_files = get_video_files(video_folder)
    if not video_files:
        print("文件夹中没有找到视频文件")
        messagebox.showinfo("提示", "文件夹中没有找到视频文件")
        return

    print(f"找到 {len(video_files)} 个视频文件")

    selected_count = select_video_count(len(video_files))
    if selected_count == 0:
        print("用户输入0，退出程序")
        return
    video_files = video_files[:selected_count]
    print(f"将处理前 {len(video_files)} 个视频")

    print("正在加载模型...")
    model = YOLO(MODEL_PATH)

    import torch
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"使用设备: {device.upper()}")
    model.to(device)

    yaml_classes = load_class_names_from_yaml(YAML_PATH)
    print("模型中的类别:", model.names)

    target_label, hao_class_id = find_target_class(yaml_classes, model.names)
    if target_label is None:
        msg = "在YAML配置文件中未找到包含 'hao' 的标签"
        print(msg)
        messagebox.showerror("错误", msg)
        return
    if hao_class_id is None:
        msg = f"模型中未找到 '{target_label}' 类别"
        print(msg)
        messagebox.showerror("错误", msg)
        return
    print(f"自动检测到目标标签: {target_label} (class_id={hao_class_id})")

    print("\n请为每个视频画起始线...")
    video_lines = {}
    for video_path in video_files:
        video_name = os.path.basename(video_path)
        cap = cv2.VideoCapture(video_path)
        ret, first_frame = cap.read()
        cap.release()
        if not ret:
            print(f"无法读取视频: {video_name}")
            continue
        print(f"请为 {video_name} 画起始线...")
        line = get_line_from_user(first_frame, video_name)
        if line is None:
            print(f"未为 {video_name} 画线，跳过")
            continue
        video_lines[video_path] = line
        print(f"已记录 {video_name} 的线: {line}")

    if not video_lines:
        print("没有视频需要处理")
        return

    os.makedirs(OUTPUT_BASE_DIR, exist_ok=True)
    print(f"\n开始处理 {len(video_lines)} 个视频...")
    results = []

    for i, (video_path, line) in enumerate(video_lines.items(), 1):
        print(f"\n{'=' * 60}\n处理进度: {i}/{len(video_lines)}")
        result = process_single_video(model, video_path, line, hao_class_id, yaml_classes, OUTPUT_BASE_DIR, device)
        if result:
            results.append(result)

    print(f"\n{'=' * 60}\n所有视频处理完成！成功处理: {len(results)}/{len(video_lines)} 个视频")
    for video_path, (output_path, report_count) in zip(video_lines.keys(), results):
        print(f"  {os.path.basename(video_path)}: Report = {report_count}\n    输出: {output_path}")

    messagebox.showinfo("完成", f"所有视频处理完成！\n成功处理: {len(results)}/{len(video_lines)} 个视频")


if __name__ == '__main__':
    main()
