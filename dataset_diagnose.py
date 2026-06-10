"""
数据集诊断脚本
功能：
  1. 类别统计 — 每个类别有多少张图、多少个标注框
  2. 模糊度检测 — Laplacian方差法，自动筛选模糊/低质图片
  3. 空标注检查 — 找出无检测对象的纯背景图
  4. 缺失标签检查 — 有图无标/有标无图的异常
  5. 输出汇总报告 + 可视化分布

用法：python dataset_diagnose.py
"""

import os
import cv2
import numpy as np
import yaml
from collections import defaultdict

# ======================== 配置区 ========================
DATASET_DIR = ""             # 数据集根目录(包含 images/labels 子目录)
YAML_PATH = ""               # 类别配置文件路径
BLUR_THRESHOLD = 100.0       # 模糊度阈值(Laplacian方差低于此值视为模糊)
REPORT_FILE = ""             # 输出报告文件路径
IMAGE_EXTENSIONS = {'.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.webp'}
# ======================================================


def get_image_files(img_dir):
    """获取目录下所有图片文件(按名称排序)"""
    return sorted(
        f for f in os.listdir(img_dir)
        if os.path.splitext(f)[1].lower() in IMAGE_EXTENSIONS
    )


def parse_yolo_label(label_path):
    """解析YOLO格式标签文件，返回 [(class_id, cx, cy, w, h), ...]"""
    if not os.path.exists(label_path):
        return None
    boxes = []
    with open(label_path, 'r', encoding='utf-8') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 5:
                boxes.append((int(parts[0]), float(parts[1]), float(parts[2]),
                              float(parts[3]), float(parts[4])))
    return boxes if boxes else []  # 空列表=有文件但无标注


def compute_laplacian_variance(img_path):
    """计算图片的Laplacian方差(清晰度指标)，无法读取返回-1"""
    img = cv2.imread(img_path)
    if img is None:
        return -1
    return cv2.Laplacian(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), cv2.CV_64F).var()


def find_label_path(img_name_no_ext, img_ext, label_index):
    """根据图片名查找对应标签文件路径"""
    # 策略1: 精确匹配(图片名.txt)
    label_name = img_name_no_ext + '.txt'
    if label_name in label_index:
        return label_index[label_name]
    # 策略2: 用文件名部分匹配
    for lname, lpath in label_index.items():
        l_base = os.path.splitext(lname)[0]
        if l_base.endswith(img_name_no_ext) or img_name_no_ext.endswith(l_base):
            return lpath
    return None


def diagnose_split(split_name, img_dir, label_dir, class_names):
    """对 train 或 val 单个子集进行完整诊断"""
    print(f"\n{'=' * 70}")
    print(f"  正在诊断 [{split_name}] 集...")
    print(f"{'=' * 70}")

    img_files = get_image_files(img_dir)
    total_images = len(img_files)

    # 统计容器
    class_counts = defaultdict(int)       # 每类图片数
    class_box_counts = defaultdict(int)   # 每类bbox总数
    empty_label_images = []               # 空标注图(背景图)
    missing_label_images = []             # 缺少标签文件的图
    unreadable_images = []                # 无法读取的图片
    blur_scores = {}                      # 图片名 -> 模糊度分数
    blurry_images = []                    # 模糊图列表

    # 建立标签索引
    label_index = {}
    if os.path.exists(label_dir):
        for lf in os.listdir(label_dir):
            if lf.endswith('.txt'):
                label_index[lf] = os.path.join(label_dir, lf)

    print(f"  图片总数: {total_images}")
    print(f"  标签文件总数: {len(label_index)}")

    # 逐张处理
    for i, img_file in enumerate(img_files):
        img_path = os.path.join(img_dir, img_file)
        img_name_no_ext = os.path.splitext(img_file)[0]

        # 查找对应标签
        label_path = find_label_path(img_name_no_ext, os.path.splitext(img_file)[1], label_index)

        if label_path is None:
            missing_label_images.append(img_file)
            continue

        # 解析标签
        boxes = parse_yolo_label(label_path)
        if boxes is None:
            missing_label_images.append(img_file)
            continue

        if len(boxes) == 0:
            empty_label_images.append(img_file)

        # 统计类别
        seen_classes = set()
        for cls_id, *_ in boxes:
            cls_name = class_names.get(cls_id, f"未知类{cls_id}")
            class_box_counts[cls_name] += 1
            seen_classes.add(cls_id)
        for cls_id in seen_classes:
            class_counts[class_names.get(cls_id, f"未知类{cls_id}")] += 1

        # 模糊度检测
        score = compute_laplacian_variance(img_path)
        if score < 0:
            unreadable_images.append(img_file)
        else:
            blur_scores[img_file] = score
            if score < BLUR_THRESHOLD:
                blurry_images.append((img_file, score))

        # 进度显示
        if (i + 1) % 500 == 0 or (i + 1) == total_images:
            print(f"  已处理: {i + 1}/{total_images} ({(i + 1) * 100 // total_images}%)")

    return {
        'split': split_name,
        'total_images': total_images,
        'class_counts': dict(class_counts),
        'class_box_counts': dict(class_box_counts),
        'empty_labels': empty_label_images,
        'missing_labels': missing_label_images,
        'unreadable': unreadable_images,
        'blur_scores': blur_scores,
        'blurry_images': blurry_images,
        'label_index_size': len(label_index),
    }


def print_report(results, class_names):
    """打印并写入诊断报告"""
    lines = []

    def log(msg=''):
        print(msg)
        lines.append(msg)

    log("\n" + "=" * 70)
    log("          数据集诊断报告")
    log("=" * 70)

    for res in results:
        split = res['split']
        log(f"\n{'─' * 70}")
        log(f"  【{split} 集】")
        log(f"{'─' * 70}")

        total = res['total_images']
        valid = total - len(res['missing_labels']) - len(res['unreadable'])
        has_label = valid - len(res['empty_labels'])

        log(f"\n  [基础信息]")
        log(f"    图片总数:       {total}")
        log(f"    标签文件数:     {res['label_index_size']}")
        log(f"    无法读取的图片: {len(res['unreadable'])}")
        log(f"    缺少标签的图片: {len(res['missing_labels'])}")
        log(f"    有效带标注图片: {has_label}")

        # 空标注
        log(f"\n  [空标注/背景图] 共 {len(res['empty_labels'])} 张")
        if res['empty_labels']:
            log(f"    （这些是没有bbox的纯背景图，建议删除）")
            for f in res['empty_labels'][:20]:
                log(f"      - {f}")
            if len(res['empty_labels']) > 20:
                log(f"      ... 还有 {len(res['empty_labels']) - 20} 张")

        # 缺少标签
        log(f"\n  [缺少标签] 共 {len(res['missing_labels'])} 张")
        if res['missing_labels']:
            for f in res['missing_labels'][:20]:
                log(f"      - {f}")
            if len(res['missing_labels']) > 20:
                log(f"      ... 还有 {len(res['missing_labels']) - 20} 张")

        # 类别分布
        log(f"\n  [类别分布]")
        log(f"    {'类别名称':<15} {'图片数':>8} {'占比':>8} {'标注框数':>10}")
        log(f"    {'-' * 15} {'-' * 8} {'-' * 8} {'-' * 10}")
        for cls_name in class_names.values():
            cnt = res['class_counts'].get(cls_name, 0)
            box_cnt = res['class_box_counts'].get(cls_name, 0)
            pct = f"{cnt * 100 / max(valid, 1):.1f}%"
            log(f"    {cls_name:<15} {cnt:>8} {pct:>8} {box_cnt:>10}")

        # 模糊度统计
        scores = list(res['blur_scores'].values())
        if scores:
            scores_arr = np.array(scores)
            log(f"\n  [模糊度分析 (Laplacian方差)]")
            log(f"    图片数(已计算): {len(scores)}")
            log(f"    平均值:         {scores_arr.mean():.1f}")
            log(f"    中位数:         {np.median(scores_arr):.1f}")
            log(f"    最小值:         {scores_arr.min():.1f}")
            log(f"    最大值:         {scores_arr.max():.1f}")
            log(f"    标准差:         {scores_arr.std():.1f}")
            log(f"    当前阈值(<{BLUR_THRESHOLD}): 判定为模糊")

            # 分段统计
            very_blurry = sum(1 for s in scores if s < BLUR_THRESHOLD)
            somewhat_blurry = sum(1 for s in scores if BLUR_THRESHOLD <= s < BLUR_THRESHOLD * 2)
            clear = sum(1 for s in scores if s >= BLUR_THRESHOLD * 2)
            log(f"\n    分布:")
            log(f"      模糊     (< {BLUR_THRESHOLD:<6}): {very_blurry:<6} 张 ({very_blurry * 100 // len(scores)}%)")
            log(f"      一般     ({BLUR_THRESHOLD:<6} ~ {int(BLUR_THRESHOLD * 2):<6}): {somewhat_blurry:<6} 张 ({somewhat_blurry * 100 // len(scores)}%)")
            log(f"      清晰     (>= {int(BLUR_THRESHOLD * 2):<6}): {clear:<6} 张 ({clear * 100 // len(scores)}%)")

            log(f"\n  [模糊图片 TOP 50] (按模糊度升序)")
            for f, s in sorted(res['blurry_images'], key=lambda x: x[1])[:50]:
                log(f"      分数={s:>8.1f}  |  {f}")

    # 总结与建议
    log(f"\n\n{'=' * 70}")
    log("  总结与优化建议")
    log(f"{'=' * 70}")

    for res in results:
        split = res['split']
        empty_n = len(res['empty_labels'])
        miss_n = len(res['missing_labels'])
        blur_n = len(res['blurry_images'])

        log(f"\n  【{split}集】建议操作:")
        log(f"    1. 删除空标注图(背景图): {empty_n} 张")
        log(f"       → 这些图没有目标对象，会干扰训练")

        if miss_n > 0:
            log(f"    2. 补充或删除缺标签图: {miss_n} 张")
            log(f"       → 有图无标，需要补标或移除")

        log(f"    3. 复核模糊/低质图: {blur_n} 张")
        log(f"       → Laplacian方差 < {BLUR_THRESHOLD}，建议人工抽查后决定是否剔除")

        # 类别不平衡检查
        if res['class_counts']:
            counts = list(res['class_counts'].values())
            if max(counts) > 0 and min(counts) > 0:
                ratio = max(counts) / min(counts)
                if ratio > 3:
                    log(f"    ⚠ 类别不平衡! 最大/最小比例 = {ratio:.1f}:1")
                    log(f"       建议为少数类补充样本，或使用加权损失")

    if REPORT_FILE:
        with open(REPORT_FILE, 'w', encoding='utf-8') as f:
            f.write('\n'.join(lines))
        log(f"\n  报告已保存至: {REPORT_FILE}")

    return lines


def main():
    # 加载类别配置
    with open(YAML_PATH, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    class_names = config.get('names', {})
    if isinstance(class_names, list):
        class_names = {i: name for i, name in enumerate(class_names)}
    print(f"类别定义: {class_names}")

    # 分别诊断 train 和 val
    results = []
    for split in ['train', 'val']:
        img_dir = os.path.join(DATASET_DIR, 'images', split)
        label_dir = os.path.join(DATASET_DIR, 'labels', split)
        if not os.path.exists(img_dir):
            print(f"警告: 图片目录不存在: {img_dir}")
            continue
        results.append(diagnose_split(split, img_dir, label_dir, class_names))

    # 生成报告
    print_report(results, class_names)


if __name__ == '__main__':
    main()
