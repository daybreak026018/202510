"""
Compact blue-white YOLO Studio panels.
"""

from __future__ import annotations

import re
import subprocess
from datetime import datetime
from pathlib import Path

import yaml

from PyQt5.QtCore import QProcess, QProcessEnvironment, QSettings, QSize, Qt
from PyQt5.QtGui import QDesktopServices, QIcon, QPixmap
from PyQt5.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFileDialog,
    QFrame,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QMessageBox,
    QPlainTextEdit,
    QPushButton,
    QSpinBox,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from PyQt5.QtCore import QUrl

from ..core.yolo_services import (
    CondaEnvironmentManager,
    EnvironmentChecker,
    IMAGE_EXTENSIONS,
    RunsManager,
    YOLOCommandBuilder,
    YOLODatasetInspector,
    YOLOPredictConfig,
    YOLOTrainConfig,
)
from ..utils.worker_thread import WorkerThread


def _button(text: str, handler=None, button_type: str = "", height: int = 36) -> QPushButton:
    btn = QPushButton(text)
    btn.setMinimumHeight(height)
    if button_type:
        btn.setProperty("buttonType", button_type)
    if handler:
        btn.clicked.connect(handler)
    return btn


def _path_row(label_text: str, line_edit: QLineEdit, button: QPushButton) -> QHBoxLayout:
    row = QHBoxLayout()
    row.setSpacing(8)
    label = QLabel(label_text)
    label.setMinimumWidth(82)
    row.addWidget(label)
    row.addWidget(line_edit, 1)
    row.addWidget(button)
    return row


class YOLOBasePanel(QWidget):
    def apply_theme(self):
        self.setStyleSheet(
            """
            QWidget {
                background-color: transparent;
                color: #222222;
                font-family: "Microsoft YaHei UI", "Segoe UI", "Microsoft YaHei";
                font-size: 12px;
            }

            QLabel#banner {
                background-color: #f5f5f5;
                border: 1px solid #cccccc;
                border-radius: 0px;
                padding: 9px 11px;
                color: #444444;
            }

            QFrame#metricCard {
                background-color: #ffffff;
                border: 1px solid #cccccc;
                border-radius: 0px;
            }

            QLabel#metricTitle {
                color: #666666;
                font-size: 10px;
                background-color: transparent;
            }

            QLabel#metricValue {
                color: #222222;
                font-size: 16px;
                font-weight: bold;
                background-color: transparent;
            }

            QGroupBox {
                background-color: #ffffff;
                border: 1px solid #cccccc;
                border-radius: 0px;
                margin-top: 10px;
                padding: 14px 10px 10px 10px;
                font-size: 12px;
                font-weight: bold;
                color: #333333;
            }

            QGroupBox::title {
                subcontrol-origin: margin;
                left: 12px;
                padding: 0 7px;
                background-color: #ffffff;
                color: #333333;
            }

            QTabWidget::pane {
                border: 1px solid #cccccc;
                border-radius: 0px;
                background-color: #ffffff;
                top: -1px;
            }

            QTabBar::tab {
                background-color: #eeeeee;
                border: 1px solid #cccccc;
                border-bottom: none;
                border-top-left-radius: 0px;
                border-top-right-radius: 0px;
                color: #555555;
                padding: 6px 11px;
                margin-right: 2px;
            }

            QTabBar::tab:selected {
                background-color: #ffffff;
                color: #222222;
                font-weight: bold;
                border-top: 2px solid #3f6fae;
            }

            QLineEdit,
            QPlainTextEdit,
            QListWidget,
            QComboBox,
            QSpinBox,
            QDoubleSpinBox {
                border: 1px solid #bdbdbd;
                border-radius: 0px;
                background-color: #ffffff;
                padding: 5px 7px;
                selection-background-color: #dbe5f2;
                selection-color: #222222;
            }

            QLineEdit:focus,
            QPlainTextEdit:focus,
            QComboBox:focus,
            QSpinBox:focus,
            QDoubleSpinBox:focus {
                border-color: #6d8fb8;
            }

            QComboBox::drop-down {
                width: 26px;
                border: none;
            }

            QComboBox QAbstractItemView {
                background-color: #ffffff;
                border: 1px solid #bdbdbd;
                selection-background-color: #dbe5f2;
                selection-color: #222222;
            }

            QPushButton {
                min-height: 30px;
                padding: 5px 12px;
                border-radius: 0px;
                border: 1px solid #aaaaaa;
                background-color: #ffffff;
                color: #222222;
                font-weight: normal;
            }

            QPushButton:hover {
                background-color: #eeeeee;
                border-color: #777777;
            }

            QPushButton:pressed {
                background-color: #dddddd;
            }

            QPushButton[buttonType="primary"],
            QPushButton[buttonType="success"] {
                background-color: #e3edf8;
                border-color: #6d8fb8;
                color: #244d7e;
            }

            QPushButton[buttonType="primary"]:hover,
            QPushButton[buttonType="success"]:hover {
                background-color: #d5e3f2;
                border-color: #4e729e;
            }

            QPushButton[buttonType="primary"]:pressed,
            QPushButton[buttonType="success"]:pressed {
                background-color: #c6d8eb;
            }

            QPushButton[buttonType="warning"] {
                background-color: #f7f0df;
                border-color: #c6a85d;
                color: #6b5421;
            }

            QPushButton[buttonType="danger"] {
                background-color: #f8e3e3;
                border-color: #c77a7a;
                color: #7c2929;
            }

            QCheckBox {
                spacing: 6px;
                color: #444444;
            }

            QCheckBox::indicator {
                width: 14px;
                height: 14px;
                border: 1px solid #999999;
                border-radius: 0px;
                background-color: #ffffff;
            }

            QCheckBox::indicator:checked {
                background-color: #6d8fb8;
                border-color: #5d7fa8;
            }

            QScrollBar:vertical {
                background-color: #eeeeee;
                width: 10px;
                border-radius: 0px;
            }

            QScrollBar::handle:vertical {
                background-color: #bdbdbd;
                border-radius: 0px;
                min-height: 24px;
            }

            QScrollBar::add-line:vertical,
            QScrollBar::sub-line:vertical {
                height: 0px;
            }

            QScrollBar:horizontal {
                background-color: #eeeeee;
                height: 10px;
                border-radius: 0px;
            }

            QScrollBar::handle:horizontal {
                background-color: #bdbdbd;
                border-radius: 0px;
                min-width: 24px;
            }

            QScrollBar::add-line:horizontal,
            QScrollBar::sub-line:horizontal {
                width: 0px;
            }
            """
        )

    def _metric_card(self, title: str, value: str = "0"):
        card = QFrame()
        card.setObjectName("metricCard")
        layout = QVBoxLayout(card)
        layout.setContentsMargins(12, 10, 12, 10)
        layout.setSpacing(4)
        title_label = QLabel(title)
        title_label.setObjectName("metricTitle")
        value_label = QLabel(value)
        value_label.setObjectName("metricValue")
        value_label.setWordWrap(True)
        layout.addWidget(title_label)
        layout.addWidget(value_label)
        return card, value_label

    def _open_path(self, path_text: str):
        if not path_text:
            return
        path = Path(path_text)
        if path.exists():
            QDesktopServices.openUrl(QUrl.fromLocalFile(str(path)))

    def _default_output_root(self) -> Path:
        settings = QSettings("DataForge", "YOLOStudio")
        saved = settings.value("default_output_dir", "")
        if saved:
            return Path(saved)
        return Path(__file__).resolve().parents[3] / "runs"

    def _workspace_root(self) -> Path:
        return Path(__file__).resolve().parents[3]


class ImagePreviewList(QListWidget):
    def __init__(self, empty_text: str, icon_size: int = 152, parent=None):
        super().__init__(parent)
        self.empty_text = empty_text
        self.icon_edge = icon_size
        self.setViewMode(QListWidget.IconMode)
        self.setResizeMode(QListWidget.Adjust)
        self.setMovement(QListWidget.Static)
        self.setSpacing(10)
        self.setIconSize(QSize(icon_size, icon_size))
        self.setMinimumHeight(icon_size + 52)
        self.setWordWrap(True)

    def set_images(self, paths):
        self.clear()
        valid_paths = [Path(path) for path in paths if path and Path(path).exists()]
        if not valid_paths:
            item = QListWidgetItem(self.empty_text)
            item.setFlags(Qt.NoItemFlags)
            self.addItem(item)
            return
        for path in valid_paths:
            pixmap = QPixmap(str(path))
            if pixmap.isNull():
                continue
            icon = pixmap.scaled(
                self.icon_edge,
                self.icon_edge,
                Qt.KeepAspectRatio,
                Qt.SmoothTransformation,
            )
            item = QListWidgetItem(path.name)
            item.setToolTip(str(path))
            item.setData(Qt.UserRole, str(path))
            item.setIcon(QIcon(icon))
            self.addItem(item)


class YOLOHomePanel(YOLOBasePanel):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.metrics = {}
        self._build_ui()
        self.refresh()

    def _build_ui(self):
        root = QVBoxLayout(self)
        root.setContentsMargins(6, 6, 6, 6)
        root.setSpacing(12)

        banner = QLabel("YOLO Studio：数据准备、环境检测、训练、检测和结果管理集中到一条工作流。")
        banner.setObjectName("banner")
        banner.setWordWrap(True)
        root.addWidget(banner)

        metrics = QGridLayout()
        metrics.setHorizontalSpacing(10)
        metrics.setVerticalSpacing(10)
        for index, key in enumerate(["环境状态", "最近训练", "最近检测", "可用权重"]):
            card, label = self._metric_card(key, "-")
            self.metrics[key] = label
            metrics.addWidget(card, index // 2, index % 2)
        root.addLayout(metrics)

        action_group = QGroupBox("快捷入口")
        actions = QHBoxLayout(action_group)
        actions.setContentsMargins(12, 16, 12, 12)
        actions.setSpacing(8)
        actions.addWidget(_button("刷新概览", self.refresh, "success"))
        actions.addWidget(_button("打开 runs", lambda: self._open_path(str(Path.cwd() / "runs"))))
        actions.addStretch()
        root.addWidget(action_group)

        self.summary = QPlainTextEdit()
        self.summary.setReadOnly(True)
        self.summary.setMinimumHeight(160)
        root.addWidget(self.summary)
        root.addStretch()

    def refresh(self):
        report = EnvironmentChecker.check()
        runs = RunsManager(Path.cwd() / "runs").list_runs()
        train_runs = [item for item in runs if item["type"] == "训练"]
        predict_runs = [item for item in runs if item["type"] == "检测"]
        weights = [item for item in runs if item.get("best") or item.get("last")]
        self.metrics["环境状态"].setText(report.status)
        self.metrics["最近训练"].setText(train_runs[0]["modified"] if train_runs else "暂无")
        self.metrics["最近检测"].setText(predict_runs[0]["modified"] if predict_runs else "暂无")
        self.metrics["可用权重"].setText(str(len(weights)))
        self.summary.setPlainText(
            "\n".join(
                [
                    f"Python: {report.python_version}",
                    f"PyTorch: {report.torch_version}",
                    f"CUDA: {report.cuda_version}",
                    f"GPU: {report.gpu_name}",
                    f"Ultralytics: {report.ultralytics_version}",
                    f"建议: {'可以开始训练' if report.ultralytics_installed else '请先到环境检测页安装依赖'}",
                ]
            )
        )


class YOLODataPanel(YOLOBasePanel):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.inspector = YOLODatasetInspector()
        self._build_ui()

    def _build_ui(self):
        root = QVBoxLayout(self)
        root.setContentsMargins(6, 6, 6, 6)
        root.setSpacing(10)

        banner = QLabel("准备 YOLO 数据集：检测/分割使用 images 与 labels，分类使用 train/val/test/类别名目录。")
        banner.setObjectName("banner")
        banner.setWordWrap(True)
        root.addWidget(banner)

        self.tabs = QTabWidget()
        self.tabs.setObjectName("dataPrepTabs")
        root.addWidget(self.tabs, 1)

        self.tabs.addTab(self._build_yolo_check_tab(), "YOLO体检")
        self.tabs.addTab(self._wrap_legacy_panel("converter"), "格式转换")
        self.tabs.addTab(self._wrap_legacy_panel("split"), "数据划分")
        self.tabs.addTab(DatasetIntegrityPanel(self), "数据集完整性检查")
        self.tabs.addTab(LabelClassPreviewPanel(self), "标签类别数量预览")

    def _build_yolo_check_tab(self):
        tab = QWidget()
        root = QVBoxLayout(tab)
        root.setContentsMargins(6, 6, 6, 6)
        root.setSpacing(10)

        group = QGroupBox("数据集")
        layout = QVBoxLayout(group)
        layout.setContentsMargins(12, 16, 12, 12)
        layout.setSpacing(8)
        self.dataset_edit = QLineEdit()
        self.dataset_edit.setReadOnly(True)
        layout.addLayout(_path_row("数据目录", self.dataset_edit, _button("选择", self.choose_dataset)))
        task_row = QHBoxLayout()
        task_row.addWidget(QLabel("任务类型"))
        self.task_combo = QComboBox()
        self.task_combo.addItems(["detect", "segment", "classify"])
        task_row.addWidget(self.task_combo, 1)
        layout.addLayout(task_row)
        root.addWidget(group)

        actions = QHBoxLayout()
        actions.addWidget(_button("扫描数据集", self.scan_dataset, "success"))
        actions.addWidget(_button("生成/修复 data.yaml", self.generate_yaml))
        actions.addWidget(_button("打开目录", lambda: self._open_path(self.dataset_edit.text())))
        actions.addStretch()
        root.addLayout(actions)

        self.report = QPlainTextEdit()
        self.report.setReadOnly(True)
        self.report.setMinimumHeight(240)
        root.addWidget(self.report)
        root.addStretch()
        return tab

    def _wrap_legacy_panel(self, panel_name: str):
        tab = QWidget()
        layout = QVBoxLayout(tab)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        if panel_name == "converter":
            from .converter_panel import ConverterPanel

            panel = ConverterPanel(self)
        elif panel_name == "split":
            from .simple_splitting_panel import SimpleSplittingPanel

            panel = SimpleSplittingPanel(self)
        else:
            panel = QLabel("未配置的页面")

        if hasattr(panel, "apply_theme"):
            panel.apply_theme()
        layout.addWidget(panel)
        return tab

    def choose_dataset(self):
        path = QFileDialog.getExistingDirectory(self, "选择 YOLO 数据集目录", str(Path.cwd()))
        if path:
            self.dataset_edit.setText(path)

    def scan_dataset(self):
        if not self.dataset_edit.text():
            QMessageBox.warning(self, "提示", "请先选择数据集目录。")
            return
        profile = self.inspector.inspect(Path(self.dataset_edit.text()), self.task_combo.currentText())
        self.report.setPlainText(self._format_profile(profile))

    def generate_yaml(self):
        if not self.dataset_edit.text():
            QMessageBox.warning(self, "提示", "请先选择数据集目录。")
            return
        if self.task_combo.currentText() == "classify":
            QMessageBox.information(self, "提示", "分类任务通常直接选择数据集根目录，不需要 data.yaml。")
            return
        path = self.inspector.generate_data_yaml(Path(self.dataset_edit.text()), self.task_combo.currentText())
        self.report.appendPlainText(f"\ndata.yaml 已生成：{path}")

    def _format_profile(self, profile):
        lines = [
            "YOLO 数据集体检",
            f"任务类型: {profile.task}",
            f"目录: {profile.dataset_path}",
            f"data.yaml: {profile.data_yaml or '缺失'}",
            f"类别数: {len(profile.classes) or profile.classification_classes}",
            f"类别: {', '.join(profile.classes[:20]) if profile.classes else '未读取到'}",
            f"train 图像: {profile.train_images}",
            f"val 图像: {profile.val_images}",
            f"test 图像: {profile.test_images}",
            f"标注文件: {profile.label_files}",
            f"空标注: {profile.empty_labels}",
            f"图像缺标注: {profile.orphan_images}",
            f"标注缺图像: {profile.orphan_labels}",
            f"异常标注行: {profile.invalid_yolo_rows}",
        ]
        if profile.warnings:
            lines.append("")
            lines.append("提醒")
            lines.extend(f"- {item}" for item in profile.warnings)
        return "\n".join(lines)


class DatasetIntegrityPanel(YOLOBasePanel):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.dataset_path = None
        self.metric_labels = {}
        self._build_ui()

    def _build_ui(self):
        root = QVBoxLayout(self)
        root.setContentsMargins(6, 6, 6, 6)
        root.setSpacing(10)

        group = QGroupBox("检查目录")
        layout = QVBoxLayout(group)
        layout.setContentsMargins(12, 16, 12, 12)
        self.path_edit = QLineEdit()
        self.path_edit.setReadOnly(True)
        layout.addLayout(_path_row("数据目录", self.path_edit, _button("选择", self.choose_dataset)))
        root.addWidget(group)

        metrics = QGridLayout()
        metrics.setHorizontalSpacing(10)
        metrics.setVerticalSpacing(10)
        for index, key in enumerate(["图片总数", "标签总数", "完整对齐", "问题数量"]):
            card, label = self._metric_card(key, "0")
            self.metric_labels[key] = label
            metrics.addWidget(card, index // 2, index % 2)
        root.addLayout(metrics)

        actions = QHBoxLayout()
        actions.addWidget(_button("开始完整性检查", self.scan, "success"))
        actions.addWidget(_button("打开目录", lambda: self._open_path(self.path_edit.text())))
        actions.addStretch()
        root.addLayout(actions)

        self.report = QPlainTextEdit()
        self.report.setReadOnly(True)
        self.report.setMinimumHeight(220)
        root.addWidget(self.report)

    def choose_dataset(self):
        path = QFileDialog.getExistingDirectory(self, "选择 YOLO 数据集目录", str(Path.cwd()))
        if path:
            self.dataset_path = Path(path)
            self.path_edit.setText(path)

    def scan(self):
        if not self.path_edit.text():
            QMessageBox.warning(self, "提示", "请先选择数据集目录。")
            return
        stats = self._scan_integrity(Path(self.path_edit.text()))
        self.metric_labels["图片总数"].setText(str(stats["images"]))
        self.metric_labels["标签总数"].setText(str(stats["labels"]))
        self.metric_labels["完整对齐"].setText(str(stats["paired"]))
        self.metric_labels["问题数量"].setText(str(stats["issues"]))
        self.report.setPlainText(self._format_integrity(stats))

    def _scan_integrity(self, root: Path):
        image_root = root / "images"
        label_root = root / "labels"
        image_files = self._collect_images(image_root if image_root.exists() else root)
        label_files = self._collect_label_files(label_root if label_root.exists() else root)
        image_keys = {self._relative_key(path, image_root, root) for path in image_files}
        label_keys = {self._relative_key(path, label_root, root) for path in label_files}
        missing_labels = sorted(image_keys - label_keys)
        missing_images = sorted(label_keys - image_keys)
        empty_labels = [str(path) for path in label_files if path.stat().st_size == 0]
        invalid_rows = []
        for label in label_files:
            invalid_rows.extend(self._invalid_rows(label))
        split_counts = {
            split: {
                "images": len([p for p in image_files if split in p.parts]),
                "labels": len([p for p in label_files if split in p.parts]),
            }
            for split in ("train", "val", "test")
        }
        issues = len(missing_labels) + len(missing_images) + len(empty_labels) + len(invalid_rows)
        return {
            "images": len(image_files),
            "labels": len(label_files),
            "paired": len(image_keys & label_keys),
            "missing_labels": missing_labels,
            "missing_images": missing_images,
            "empty_labels": empty_labels,
            "invalid_rows": invalid_rows,
            "split_counts": split_counts,
            "issues": issues,
        }

    def _collect_images(self, root: Path):
        if not root.exists():
            return []
        return [path for path in root.rglob("*") if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS]

    def _collect_label_files(self, root: Path):
        ignored = {"classes.txt", "labels.txt", "label_map.txt", "label_mapping.txt"}
        if not root.exists():
            return []
        return [path for path in root.rglob("*.txt") if path.name.lower() not in ignored]

    def _relative_key(self, path: Path, preferred_root: Path, fallback_root: Path):
        try:
            base = preferred_root if preferred_root.exists() else fallback_root
            return str(path.relative_to(base).with_suffix("")).replace("\\", "/")
        except ValueError:
            return path.stem

    def _invalid_rows(self, label_path: Path):
        invalid = []
        for line_no, line in enumerate(label_path.read_text(encoding="utf-8", errors="ignore").splitlines(), 1):
            parts = line.split()
            if not parts:
                continue
            if len(parts) < 5:
                invalid.append(f"{label_path}:{line_no} 字段不足")
                continue
            try:
                int(float(parts[0]))
                values = [float(item) for item in parts[1:]]
            except ValueError:
                invalid.append(f"{label_path}:{line_no} 非数字字段")
                continue
            if any(value < 0 or value > 1 for value in values):
                invalid.append(f"{label_path}:{line_no} 坐标超出 0-1")
        return invalid

    def _format_integrity(self, stats):
        lines = [
            "数据集完整性检查",
            f"图片总数: {stats['images']}",
            f"标签总数: {stats['labels']}",
            f"完整对齐: {stats['paired']}",
            "",
            "子集统计",
        ]
        for split, counts in stats["split_counts"].items():
            lines.append(f"- {split}: 图片 {counts['images']} / 标签 {counts['labels']}")
        sections = [
            ("缺少标签的图片", stats["missing_labels"]),
            ("缺少图片的标签", stats["missing_images"]),
            ("空标签文件", stats["empty_labels"]),
            ("异常标签行", stats["invalid_rows"]),
        ]
        for title, items in sections:
            lines.append("")
            lines.append(f"{title}: {len(items)}")
            lines.extend(f"- {item}" for item in items[:30])
            if len(items) > 30:
                lines.append(f"... 还有 {len(items) - 30} 项")
        return "\n".join(lines)


class LabelClassPreviewPanel(YOLOBasePanel):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.metric_labels = {}
        self._build_ui()

    def _build_ui(self):
        root = QVBoxLayout(self)
        root.setContentsMargins(6, 6, 6, 6)
        root.setSpacing(10)

        group = QGroupBox("预览目录")
        layout = QVBoxLayout(group)
        layout.setContentsMargins(12, 16, 12, 12)
        self.path_edit = QLineEdit()
        self.path_edit.setReadOnly(True)
        layout.addLayout(_path_row("数据目录", self.path_edit, _button("选择", self.choose_dataset)))
        task_row = QHBoxLayout()
        task_row.addWidget(QLabel("任务类型"))
        self.task_combo = QComboBox()
        self.task_combo.addItems(["detect", "segment", "classify"])
        task_row.addWidget(self.task_combo, 1)
        layout.addLayout(task_row)
        root.addWidget(group)

        metrics = QGridLayout()
        metrics.setHorizontalSpacing(10)
        metrics.setVerticalSpacing(10)
        for index, key in enumerate(["类别数量", "标签实例", "标签文件", "图片数量"]):
            card, label = self._metric_card(key, "0")
            self.metric_labels[key] = label
            metrics.addWidget(card, index // 2, index % 2)
        root.addLayout(metrics)

        actions = QHBoxLayout()
        actions.addWidget(_button("生成类别预览", self.preview, "success"))
        actions.addWidget(_button("打开目录", lambda: self._open_path(self.path_edit.text())))
        actions.addStretch()
        root.addLayout(actions)

        self.report = QPlainTextEdit()
        self.report.setReadOnly(True)
        self.report.setMinimumHeight(240)
        root.addWidget(self.report)

    def choose_dataset(self):
        path = QFileDialog.getExistingDirectory(self, "选择 YOLO 数据集目录", str(Path.cwd()))
        if path:
            self.path_edit.setText(path)

    def preview(self):
        if not self.path_edit.text():
            QMessageBox.warning(self, "提示", "请先选择数据集目录。")
            return
        root = Path(self.path_edit.text())
        task = self.task_combo.currentText()
        stats = self._scan_classification(root) if task == "classify" else self._scan_yolo_labels(root)
        self.metric_labels["类别数量"].setText(str(stats["class_count"]))
        self.metric_labels["标签实例"].setText(str(stats["instances"]))
        self.metric_labels["标签文件"].setText(str(stats["label_files"]))
        self.metric_labels["图片数量"].setText(str(stats["images"]))
        self.report.setPlainText(self._format_distribution(stats))

    def _scan_yolo_labels(self, root: Path):
        label_root = root / "labels"
        labels = self._collect_label_files(label_root if label_root.exists() else root)
        image_root = root / "images"
        images = self._collect_images(image_root if image_root.exists() else root)
        names = self._read_class_names(root)
        counts = {}
        invalid_class_ids = []
        empty_files = 0
        for label in labels:
            lines = label.read_text(encoding="utf-8", errors="ignore").splitlines()
            if not lines:
                empty_files += 1
            for line_no, line in enumerate(lines, 1):
                parts = line.split()
                if not parts:
                    continue
                try:
                    class_id = int(float(parts[0]))
                except ValueError:
                    invalid_class_ids.append(f"{label}:{line_no} 类别ID无效")
                    continue
                counts[class_id] = counts.get(class_id, 0) + 1
        total = sum(counts.values())
        class_count = max(len(names), len(counts))
        return {
            "task": "detect/segment",
            "names": names,
            "counts": counts,
            "instances": total,
            "label_files": len(labels),
            "images": len(images),
            "class_count": class_count,
            "empty_files": empty_files,
            "invalid_class_ids": invalid_class_ids,
        }

    def _scan_classification(self, root: Path):
        counts = {}
        for split in ("train", "val", "test"):
            split_dir = root / split
            if not split_dir.exists():
                continue
            for class_dir in split_dir.iterdir():
                if not class_dir.is_dir():
                    continue
                counts[class_dir.name] = counts.get(class_dir.name, 0) + len(
                    [path for path in class_dir.rglob("*") if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS]
                )
        return {
            "task": "classify",
            "names": sorted(counts),
            "counts": counts,
            "instances": sum(counts.values()),
            "label_files": 0,
            "images": sum(counts.values()),
            "class_count": len(counts),
            "empty_files": 0,
            "invalid_class_ids": [],
        }

    def _collect_images(self, root: Path):
        if not root.exists():
            return []
        return [path for path in root.rglob("*") if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS]

    def _collect_label_files(self, root: Path):
        ignored = {"classes.txt", "labels.txt", "label_map.txt", "label_mapping.txt"}
        if not root.exists():
            return []
        return [path for path in root.rglob("*.txt") if path.name.lower() not in ignored]

    def _read_class_names(self, root: Path):
        data_yaml = root / "data.yaml"
        if data_yaml.exists():
            try:
                data = yaml.safe_load(data_yaml.read_text(encoding="utf-8")) or {}
                names = data.get("names", [])
                if isinstance(names, dict):
                    return [str(names[key]) for key in sorted(names)]
                if isinstance(names, list):
                    return [str(name) for name in names]
            except Exception:
                pass
        classes = root / "classes.txt"
        if classes.exists():
            return [line.strip() for line in classes.read_text(encoding="utf-8").splitlines() if line.strip()]
        return []

    def _format_distribution(self, stats):
        lines = [
            "标签类别数量预览",
            f"任务类型: {stats['task']}",
            f"类别数量: {stats['class_count']}",
            f"标签实例/图片样本: {stats['instances']}",
            f"标签文件: {stats['label_files']}",
            f"图片数量: {stats['images']}",
            f"空标签文件: {stats['empty_files']}",
            "",
            "类别分布",
        ]
        counts = stats["counts"]
        max_count = max(counts.values(), default=0)
        if stats["task"] == "classify":
            items = sorted(counts.items(), key=lambda item: item[0])
            for name, count in items:
                lines.append(self._bar_line(name, count, max_count, stats["instances"]))
        else:
            for class_id in sorted(counts):
                name = stats["names"][class_id] if class_id < len(stats["names"]) else f"class_{class_id}"
                lines.append(self._bar_line(f"{class_id} {name}", counts[class_id], max_count, stats["instances"]))
        if stats["invalid_class_ids"]:
            lines.append("")
            lines.append(f"异常类别ID: {len(stats['invalid_class_ids'])}")
            lines.extend(f"- {item}" for item in stats["invalid_class_ids"][:20])
        return "\n".join(lines)

    def _bar_line(self, name, count, max_count, total):
        width = 24
        filled = int(count / max_count * width) if max_count else 0
        percent = count / total * 100 if total else 0
        return f"{name}: {count} ({percent:.1f}%) {'#' * filled}"


class YOLOEnvironmentPanel(YOLOBasePanel):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.settings = QSettings("DataForge", "YOLOStudio")
        self.conda_envs = []
        self._build_ui()
        self.refresh_envs()
        self.refresh()

    def _build_ui(self):
        root = QVBoxLayout(self)
        root.setContentsMargins(6, 6, 6, 6)
        root.setSpacing(12)
        banner = QLabel("环境检测只读执行，不自动安装或修改系统环境。")
        banner.setObjectName("banner")
        banner.setWordWrap(True)
        root.addWidget(banner)

        env_group = QGroupBox("Conda / Python 环境")
        env_layout = QVBoxLayout(env_group)
        env_layout.setContentsMargins(12, 16, 12, 12)
        env_layout.setSpacing(8)
        self.env_combo = QComboBox()
        env_layout.addWidget(self.env_combo)
        env_actions = QHBoxLayout()
        env_actions.addWidget(_button("刷新环境", self.refresh_envs))
        env_actions.addWidget(_button("保存为训练/检测环境", self.save_selected_env, "success"))
        env_actions.addWidget(_button("检测当前选择", self.refresh))
        env_actions.addStretch()
        env_layout.addLayout(env_actions)
        root.addWidget(env_group)

        self.report = QPlainTextEdit()
        self.report.setReadOnly(True)
        self.report.setMinimumHeight(250)
        root.addWidget(self.report)

    def refresh_envs(self):
        selected_python = self.settings.value("selected_python", "")
        self.conda_envs = CondaEnvironmentManager.list_environments()
        self.env_combo.clear()
        selected_index = 0
        for index, env in enumerate(self.conda_envs):
            label = f"{env['name']}  |  {env['python']}"
            self.env_combo.addItem(label, env)
            if selected_python and env["python"] == selected_python:
                selected_index = index
        if self.conda_envs:
            self.env_combo.setCurrentIndex(selected_index)

    def save_selected_env(self):
        env = self.env_combo.currentData()
        if not env:
            QMessageBox.warning(self, "提示", "没有可保存的环境。")
            return
        self.settings.setValue("selected_env_name", env["name"])
        self.settings.setValue("selected_env_path", env["path"])
        self.settings.setValue("selected_python", env["python"])
        QMessageBox.information(self, "完成", f"已切换到环境：{env['name']}")
        self.refresh()

    def refresh(self):
        env = self.env_combo.currentData() if hasattr(self, "env_combo") else None
        python_executable = env["python"] if env else ""
        env_name = env["name"] if env else "当前环境"
        report = EnvironmentChecker.check(python_executable, env_name)
        saved_python = self.settings.value("selected_python", "")
        selected_note = "是" if saved_python and saved_python == report.python_executable else "否"
        self.report.setPlainText(
            "\n".join(
                [
                    f"当前选择: {report.environment_name}",
                    f"是否已保存为训练/检测环境: {selected_note}",
                    f"Python路径: {report.python_executable}",
                    f"状态: {report.status}",
                    f"系统: {report.platform}",
                    f"Python: {report.python_version}",
                    f"PyTorch: {report.torch_version}",
                    f"CUDA 可用: {'是' if report.cuda_available else '否'}",
                    f"CUDA 版本: {report.cuda_version}",
                    f"GPU: {report.gpu_name}",
                    f"Ultralytics: {report.ultralytics_version}",
                    f"OpenCV: {report.opencv_version}",
                    "",
                    "CPU 安装命令:",
                    report.cpu_install_command,
                    "",
                    "GPU 安装建议:",
                    report.gpu_install_command,
                ]
            )
        )


class YOLOProcessPanel(YOLOBasePanel):
    process_kind = "process"
    ANSI_ESCAPE_RE = re.compile(r"\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])")

    def __init__(self, parent=None):
        super().__init__(parent)
        self.process = None
        self.settings = QSettings("DataForge", "YOLOStudio")

    def _start_process(self, command: list[str], cwd: Path):
        if self.process and self.process.state() != QProcess.NotRunning:
            QMessageBox.warning(self, "提示", "已有任务正在运行。")
            return
        self.log.clear()
        self.log.appendPlainText(f"使用环境: {self._selected_env_name()}\n")
        self.log.appendPlainText("执行命令:\n" + subprocess.list2cmdline(command) + "\n")
        self.process = QProcess(self)
        self.process.setProgram(command[0])
        self.process.setArguments(command[1:])
        self.process.setWorkingDirectory(str(cwd))
        self.process.setProcessEnvironment(self._process_environment(command[0]))
        self.process.setProcessChannelMode(QProcess.MergedChannels)
        self.process.readyReadStandardOutput.connect(self._read_output)
        self.process.finished.connect(self._finished)
        self.process.start()

    def _read_output(self):
        data = bytes(self.process.readAllStandardOutput()).decode("utf-8", errors="ignore")
        if data:
            cleaned = self._clean_process_output(data)
            if cleaned:
                self.log.appendPlainText(cleaned.rstrip())

    def _finished(self, exit_code, exit_status):
        status = "完成" if exit_code == 0 else f"失败({exit_code})"
        self.log.appendPlainText(f"\n任务{status}。")

    def stop_process(self):
        if self.process and self.process.state() != QProcess.NotRunning:
            self.process.terminate()
            self.log.appendPlainText("\n已请求停止任务。")

    def _selected_python(self):
        return self.settings.value("selected_python", "")

    def _selected_env_name(self):
        return self.settings.value("selected_env_name", "当前Python")

    def _selected_env_label(self):
        python = self._selected_python()
        if python:
            return f"{self._selected_env_name()} | {python}"
        return f"{self._selected_env_name()} | 当前进程 Python"

    def _process_environment(self, python_executable: str):
        env = QProcessEnvironment.systemEnvironment()
        python_path = Path(python_executable)
        env_root = python_path.parent
        path_parts = [
            str(env_root),
            str(env_root / "Scripts"),
            str(env_root / "Library" / "bin"),
            env.value("PATH", ""),
        ]
        env.insert("PATH", ";".join(part for part in path_parts if part))
        return env

    def _clean_process_output(self, text: str):
        text = text.replace("\r\n", "\n").replace("\r", "\n")
        text = self.ANSI_ESCAPE_RE.sub("", text)
        text = text.replace("\x00", "")
        return text


class YOLOTrainingPanel(YOLOProcessPanel):
    MODEL_OPTIONS = {
        "detect": ["yolo11n", "yolo11s", "yolo11m", "yolo11l", "yolo11x", "yolov8n", "yolov8s", "yolov8m", "yolov8l", "yolov8x", "yolov5n", "yolov5s", "yolov5m", "yolov5l", "yolov5x", "yolo26n", "yolo26s", "yolo26m", "yolo26l", "yolo26x"],
        "segment": ["yolo11n-seg", "yolo11s-seg", "yolo11m-seg", "yolo11l-seg", "yolo11x-seg", "yolov8n-seg", "yolov8s-seg", "yolov8m-seg", "yolov8l-seg", "yolov8x-seg"],
        "classify": ["yolo11n-cls", "yolo11s-cls", "yolo11m-cls", "yolo11l-cls", "yolo11x-cls", "yolov8n-cls", "yolov8s-cls", "yolov8m-cls", "yolov8l-cls", "yolov8x-cls"],
    }

    def __init__(self, parent=None):
        super().__init__(parent)
        self.runs_manager = RunsManager(Path.cwd() / "runs")
        self._build_ui()

    def _build_ui(self):
        root = QVBoxLayout(self)
        root.setContentsMargins(6, 6, 6, 6)
        root.setSpacing(12)
        banner = QLabel("本机一键训练：选择任务、YOLO版本和数据后启动。支持 YOLO26 / YOLO11 / YOLOv8 多尺寸模型。勾选预训练时使用 .pt，不勾选时从 .yaml 开始训练。")
        banner.setObjectName("banner")
        banner.setWordWrap(True)
        root.addWidget(banner)

        group = QGroupBox("训练配置")
        grid = QGridLayout(group)
        grid.setContentsMargins(12, 16, 12, 12)
        grid.setHorizontalSpacing(8)
        grid.setVerticalSpacing(8)

        self.task_combo = QComboBox()
        self.task_combo.addItems(["detect", "segment", "classify"])
        self.task_combo.currentTextChanged.connect(self._refresh_model_options)
        self.model_combo = QComboBox()
        self.model_combo.setEditable(True)
        self._refresh_model_options(self.task_combo.currentText())
        self.data_edit = QLineEdit()
        self.env_edit = QLineEdit()
        self.env_edit.setReadOnly(True)
        self.env_edit.setText(self._selected_env_label())
        self.project_edit = QLineEdit(str(Path.cwd() / "runs" / "train"))
        self.name_edit = QLineEdit(self._default_run_name())
        self.epochs_spin = self._spin(1, 1000, 100)
        self.imgsz_spin = self._spin(64, 2048, 640)
        self.batch_spin = self._spin(1, 512, 16)
        self.workers_spin = self._spin(0, 32, 4)
        self.patience_spin = self._spin(0, 300, 50)
        self.device_combo = QComboBox()
        self.device_combo.setEditable(True)
        self.device_combo.addItems(["auto", "cpu", "0"])
        self.pretrained_check = QCheckBox("使用预训练权重（.pt）")
        self.pretrained_check.setChecked(True)
        self.resume_check = QCheckBox("恢复训练 resume=True")

        rows = [
            ("任务", self.task_combo),
            ("YOLO版本", self.model_combo),
            ("数据", self.data_edit),
            ("运行环境", self.env_edit),
            ("输出", self.project_edit),
            ("名称", self.name_edit),
            ("epochs", self.epochs_spin),
            ("imgsz", self.imgsz_spin),
            ("batch", self.batch_spin),
            ("device", self.device_combo),
            ("workers", self.workers_spin),
            ("patience", self.patience_spin),
        ]
        for index, (label, widget) in enumerate(rows):
            grid.addWidget(QLabel(label), index // 2, (index % 2) * 2)
            grid.addWidget(widget, index // 2, (index % 2) * 2 + 1)
        grid.addWidget(self.pretrained_check, 6, 0, 1, 2)
        grid.addWidget(self.resume_check, 6, 2, 1, 2)
        grid.addWidget(_button("选择 data.yaml/分类目录", self.choose_data), 7, 2, 1, 2)
        root.addWidget(group)

        actions = QHBoxLayout()
        actions.addWidget(_button("开始训练", self.start_training, "success"))
        actions.addWidget(_button("停止", self.stop_process))
        actions.addWidget(_button("打开输出目录", lambda: self._open_path(self.project_edit.text())))
        actions.addStretch()
        root.addLayout(actions)

        self.log = QPlainTextEdit()
        self.log.setReadOnly(True)
        self.log.setMinimumHeight(220)
        root.addWidget(self.log)

        preview_group = QGroupBox("训练曲线预览")
        preview_layout = QVBoxLayout(preview_group)
        preview_layout.setContentsMargins(12, 16, 12, 12)
        self.curve_hint = QLabel("训练完成后自动显示最近一次训练曲线。")
        self.curve_hint.setWordWrap(True)
        preview_layout.addWidget(self.curve_hint)
        self.curve_preview = QLabel("暂无曲线图")
        self.curve_preview.setAlignment(Qt.AlignCenter)
        self.curve_preview.setMinimumHeight(220)
        self.curve_preview.setObjectName("curvePreview")
        preview_layout.addWidget(self.curve_preview)
        root.addWidget(preview_group)

        self.refresh_curve_preview()

    def _refresh_model_options(self, task: str):
        current = self.model_combo.currentText().strip()
        options = self.MODEL_OPTIONS.get(task, self.MODEL_OPTIONS["detect"])
        self.model_combo.blockSignals(True)
        self.model_combo.clear()
        self.model_combo.addItems(options)
        if current:
            self.model_combo.setEditText(current)
        elif options:
            self.model_combo.setCurrentText(options[0])
        self.model_combo.blockSignals(False)

    def _spin(self, min_value, max_value, value):
        spin = QSpinBox()
        spin.setRange(min_value, max_value)
        spin.setValue(value)
        return spin

    def _default_run_name(self):
        return f"train_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

    def _resolved_model_name(self):
        version = self.model_combo.currentText().strip()
        if not version:
            version = "yolo26n"
            self.model_combo.setCurrentText(version)
        if version.endswith(".pt") or version.endswith(".yaml"):
            return version
        suffix = ".pt" if self.pretrained_check.isChecked() else ".yaml"
        return f"{version}{suffix}"

    def choose_data(self):
        if self.task_combo.currentText() == "classify":
            path = QFileDialog.getExistingDirectory(self, "选择分类数据集根目录", str(Path.cwd()))
        else:
            path, _ = QFileDialog.getOpenFileName(self, "选择 data.yaml", str(Path.cwd()), "YAML Files (*.yaml *.yml)")
        if path:
            self.data_edit.setText(path)

    def start_training(self):
        self.env_edit.setText(self._selected_env_label())
        if not self.data_edit.text():
            QMessageBox.warning(self, "提示", "请先选择训练数据。")
            return
        run_name = self.name_edit.text().strip()
        if not run_name or run_name == "exp":
            run_name = self._default_run_name()
            self.name_edit.setText(run_name)
        resolved_model = self._resolved_model_name()
        self.log.appendPlainText(
            f"模型版本: {self.model_combo.currentText().strip()} | "
            f"{'预训练权重' if self.pretrained_check.isChecked() else '从配置开始训练'} | "
            f"实际模型文件: {resolved_model}\n"
        )
        cfg = YOLOTrainConfig(
            task=self.task_combo.currentText(),
            model=resolved_model,
            data_yaml=self.data_edit.text(),
            epochs=self.epochs_spin.value(),
            imgsz=self.imgsz_spin.value(),
            batch=self.batch_spin.value(),
            device=self.device_combo.currentText(),
            project=self.project_edit.text(),
            name=run_name,
            workers=self.workers_spin.value(),
            patience=self.patience_spin.value(),
            resume=self.resume_check.isChecked(),
            python_executable=self._selected_python(),
        )
        self._set_curve_preview(None)
        self._start_process(YOLOCommandBuilder.build_train(cfg), self._workspace_root())

    def _finished(self, exit_code, exit_status):
        super()._finished(exit_code, exit_status)
        self.refresh_curve_preview()

    def refresh_curve_preview(self):
        train_runs = [item for item in self.runs_manager.list_runs() if "train" in item["path"].lower()]
        latest = train_runs[0] if train_runs else None
        curve_path = Path(latest["curve"]) if latest and latest.get("curve") else None
        self._set_curve_preview(curve_path)
        if latest:
            self.curve_hint.setText(f"最近训练: {latest['name']} | {latest['modified']}")

    def _set_curve_preview(self, curve_path):
        if not curve_path or not Path(curve_path).exists():
            self.curve_preview.setPixmap(QPixmap())
            self.curve_preview.setText("暂无曲线图")
            return
        pixmap = QPixmap(str(curve_path))
        if pixmap.isNull():
            self.curve_preview.setPixmap(QPixmap())
            self.curve_preview.setText("曲线图读取失败")
            return
        scaled = pixmap.scaled(680, 240, Qt.KeepAspectRatio, Qt.SmoothTransformation)
        self.curve_preview.setPixmap(scaled)
        self.curve_preview.setText("")


class YOLOPredictPanel(YOLOProcessPanel):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.runs_manager = RunsManager(Path.cwd() / "runs")
        self._build_ui()

    def _build_ui(self):
        root = QVBoxLayout(self)
        root.setContentsMargins(6, 6, 6, 6)
        root.setSpacing(12)
        banner = QLabel("模型检测：支持图片、文件夹、视频和摄像头编号，结果保存到指定目录。")
        banner.setObjectName("banner")
        banner.setWordWrap(True)
        root.addWidget(banner)

        group = QGroupBox("检测配置")
        grid = QGridLayout(group)
        grid.setContentsMargins(12, 16, 12, 12)
        self.task_combo = QComboBox()
        self.task_combo.addItems(["detect", "segment", "classify"])
        self.model_edit = QLineEdit()
        self.source_edit = QLineEdit()
        self.env_edit = QLineEdit()
        self.env_edit.setReadOnly(True)
        self.env_edit.setText(self._selected_env_label())
        self.output_edit = QLineEdit(str(Path.cwd() / "runs" / "predict"))
        self.device_combo = QComboBox()
        self.device_combo.setEditable(True)
        self.device_combo.addItems(["auto", "cpu", "0"])
        self.conf_spin = self._double_spin(0.01, 1.0, 0.25)
        self.iou_spin = self._double_spin(0.01, 1.0, 0.7)
        self.imgsz_spin = QSpinBox()
        self.imgsz_spin.setRange(64, 2048)
        self.imgsz_spin.setValue(640)
        self.save_txt_check = QCheckBox("保存 txt")
        self.save_conf_check = QCheckBox("保存置信度")

        rows = [
            ("任务", self.task_combo),
            ("权重", self.model_edit),
            ("输入源", self.source_edit),
            ("运行环境", self.env_edit),
            ("输出", self.output_edit),
            ("conf", self.conf_spin),
            ("iou", self.iou_spin),
            ("imgsz", self.imgsz_spin),
            ("device", self.device_combo),
        ]
        for index, (label, widget) in enumerate(rows):
            grid.addWidget(QLabel(label), index // 2, (index % 2) * 2)
            grid.addWidget(widget, index // 2, (index % 2) * 2 + 1)
        grid.addWidget(self.save_txt_check, 5, 0)
        grid.addWidget(self.save_conf_check, 5, 1)
        grid.addWidget(_button("选择权重", self.choose_model), 5, 2)
        grid.addWidget(_button("选择输入", self.choose_source), 5, 3)
        root.addWidget(group)

        actions = QHBoxLayout()
        actions.addWidget(_button("开始检测", self.start_predict, "success"))
        actions.addWidget(_button("停止", self.stop_process))
        actions.addWidget(_button("打开输出目录", lambda: self._open_path(self.output_edit.text())))
        actions.addStretch()
        root.addLayout(actions)

        self.log = QPlainTextEdit()
        self.log.setReadOnly(True)
        self.log.setMinimumHeight(220)
        root.addWidget(self.log)

        preview_group = QGroupBox("检测结果缩略图")
        preview_layout = QVBoxLayout(preview_group)
        preview_layout.setContentsMargins(12, 16, 12, 12)
        self.preview_hint = QLabel("检测完成后自动显示最近一次检测结果缩略图。")
        self.preview_hint.setWordWrap(True)
        preview_layout.addWidget(self.preview_hint)
        self.preview_list = ImagePreviewList("暂无检测结果缩略图", icon_size=144)
        self.preview_list.itemDoubleClicked.connect(self._open_preview_item)
        preview_layout.addWidget(self.preview_list)
        root.addWidget(preview_group)

        self.refresh_prediction_preview()

    def _double_spin(self, min_value, max_value, value):
        spin = QDoubleSpinBox()
        spin.setRange(min_value, max_value)
        spin.setSingleStep(0.05)
        spin.setDecimals(2)
        spin.setValue(value)
        return spin

    def choose_model(self):
        path, _ = QFileDialog.getOpenFileName(self, "选择 YOLO 权重", str(Path.cwd()), "Model Files (*.pt *.onnx *.engine);;All Files (*)")
        if path:
            self.model_edit.setText(path)

    def choose_source(self):
        path, _ = QFileDialog.getOpenFileName(
            self,
            "选择图片或视频",
            str(Path.cwd()),
            "Media Files (*.jpg *.jpeg *.png *.bmp *.webp *.mp4 *.avi *.mov *.mkv);;All Files (*)",
        )
        if not path:
            path = QFileDialog.getExistingDirectory(self, "选择图片/视频目录", str(Path.cwd()))
        if path:
            self.source_edit.setText(path)

    def start_predict(self):
        self.env_edit.setText(self._selected_env_label())
        if not self.model_edit.text() or not self.source_edit.text():
            QMessageBox.warning(self, "提示", "请先选择权重和输入源。")
            return
        cfg = YOLOPredictConfig(
            task=self.task_combo.currentText(),
            model_path=self.model_edit.text(),
            source=self.source_edit.text(),
            conf=self.conf_spin.value(),
            iou=self.iou_spin.value(),
            imgsz=self.imgsz_spin.value(),
            device=self.device_combo.currentText(),
            save_txt=self.save_txt_check.isChecked(),
            save_conf=self.save_conf_check.isChecked(),
            output_dir=self.output_edit.text(),
            python_executable=self._selected_python(),
        )
        self.preview_list.set_images([])
        self._start_process(YOLOCommandBuilder.build_predict(cfg), self._workspace_root())

    def _finished(self, exit_code, exit_status):
        super()._finished(exit_code, exit_status)
        self.refresh_prediction_preview()

    def set_model_path(self, path_text: str):
        self.model_edit.setText(path_text)

    def refresh_prediction_preview(self):
        predict_runs = [item for item in self.runs_manager.list_runs() if "predict" in item["path"].lower()]
        latest = predict_runs[0] if predict_runs else None
        preview_images = latest.get("preview_images", []) if latest else []
        self.preview_list.set_images(preview_images)
        if latest:
            self.preview_hint.setText(f"最近检测: {latest['name']} | {latest['modified']} | 双击缩略图可打开原图")

    def _open_preview_item(self, item):
        path_text = item.data(Qt.UserRole)
        if path_text:
            self._open_path(path_text)


class YOLORunsPanel(YOLOBasePanel):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.runs_root = Path.cwd() / "runs"
        self.runs = []
        self._build_ui()
        self.refresh()

    def _build_ui(self):
        root = QVBoxLayout(self)
        root.setContentsMargins(6, 6, 6, 6)
        root.setSpacing(12)
        banner = QLabel("结果管理：扫描 runs 目录，快速找到训练权重和预测输出。")
        banner.setObjectName("banner")
        banner.setWordWrap(True)
        root.addWidget(banner)
        actions = QHBoxLayout()
        actions.addWidget(_button("刷新", self.refresh, "success"))
        actions.addWidget(_button("打开选中目录", self.open_selected))
        actions.addWidget(_button("复制 best.pt 路径", self.copy_best))
        actions.addStretch()
        root.addLayout(actions)
        self.list_widget = QListWidget()
        self.list_widget.currentRowChanged.connect(self.show_detail)
        root.addWidget(self.list_widget, 1)
        self.detail = QPlainTextEdit()
        self.detail.setReadOnly(True)
        self.detail.setMinimumHeight(120)
        root.addWidget(self.detail)

    def refresh(self):
        self.runs = RunsManager(self.runs_root).list_runs()
        self.list_widget.clear()
        for item in self.runs:
            self.list_widget.addItem(f"[{item['type']}] {item['modified']}  {item['path']}")
        self.detail.setPlainText(f"共发现 {len(self.runs)} 个结果。")

    def show_detail(self, index):
        if index < 0 or index >= len(self.runs):
            return
        item = self.runs[index]
        self.detail.setPlainText(
            "\n".join(
                [
                    f"类型: {item['type']}",
                    f"目录: {item['path']}",
                    f"best.pt: {item['best'] or '无'}",
                    f"last.pt: {item['last'] or '无'}",
                    f"更新时间: {item['modified']}",
                ]
            )
        )

    def open_selected(self):
        index = self.list_widget.currentRow()
        if 0 <= index < len(self.runs):
            self._open_path(self.runs[index]["path"])

    def copy_best(self):
        index = self.list_widget.currentRow()
        if 0 <= index < len(self.runs) and self.runs[index]["best"]:
            from PyQt5.QtWidgets import QApplication

            QApplication.clipboard().setText(self.runs[index]["best"])
            QMessageBox.information(self, "完成", "best.pt 路径已复制。")


class YOLOSettingsPanel(YOLOBasePanel):
    def __init__(self, parent=None):
        super().__init__(parent)
        self._build_ui()

    def _build_ui(self):
        root = QVBoxLayout(self)
        root.setContentsMargins(6, 6, 6, 6)
        root.setSpacing(12)
        banner = QLabel("YOLO Studio 设置：第一版保留蓝白固定主题，环境安装由用户按检测页命令执行。")
        banner.setObjectName("banner")
        banner.setWordWrap(True)
        root.addWidget(banner)
        info = QPlainTextEdit()
        info.setReadOnly(True)
        info.setPlainText(
            "默认输出目录:\n"
            f"{Path.cwd() / 'runs'}\n\n"
            "推荐流程:\n"
            "1. 环境检测确认 ultralytics 与 torch 可用\n"
            "2. 数据准备扫描并生成 data.yaml\n"
            "3. 模型训练启动本机训练\n"
            "4. 模型检测选择 best.pt 运行推理\n"
            "5. 结果管理查看输出"
        )
        root.addWidget(info)
        root.addStretch()


_LegacyYOLOTrainingPanel = YOLOTrainingPanel


class YOLOTrainingPanel(_LegacyYOLOTrainingPanel):
    def refresh_curve_preview(self):
        train_runs = [item for item in self.runs_manager.list_runs() if "train" in item["path"].lower()]
        latest = train_runs[0] if train_runs else None
        curve_path = Path(latest["curve"]) if latest and latest.get("curve") else None
        self._set_curve_preview(curve_path)
        if latest:
            self.curve_hint.setText(f"最近训练: {latest['name']} | {latest['modified']}")
        else:
            self.curve_hint.setText("训练完成后自动显示最近一次训练曲线。")


_LegacyYOLOPredictPanel = YOLOPredictPanel


class YOLOPredictPanel(_LegacyYOLOPredictPanel):
    def refresh_prediction_preview(self):
        predict_runs = [item for item in self.runs_manager.list_runs() if "predict" in item["path"].lower()]
        latest = predict_runs[0] if predict_runs else None
        preview_images = latest.get("preview_images", []) if latest else []
        self.preview_list.set_images(preview_images)
        if latest:
            self.preview_hint.setText(f"最近检测: {latest['name']} | {latest['modified']} | 双击缩略图可打开原图")
        else:
            self.preview_hint.setText("检测完成后自动显示最近一次检测结果缩略图。")


_LegacyYOLORunsPanel = YOLORunsPanel


class YOLORunsPanel(YOLOBasePanel):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.runs_root = Path.cwd() / "runs"
        self.runs = []
        self._build_ui()
        self.refresh()

    def _build_ui(self):
        root = QVBoxLayout(self)
        root.setContentsMargins(6, 6, 6, 6)
        root.setSpacing(12)

        banner = QLabel("结果管理：集中查看训练曲线、检测缩略图和权重文件。")
        banner.setObjectName("banner")
        banner.setWordWrap(True)
        root.addWidget(banner)

        actions = QHBoxLayout()
        actions.addWidget(_button("刷新", self.refresh, "success"))
        actions.addWidget(_button("打开选中目录", self.open_selected))
        actions.addWidget(_button("复制 best.pt 路径", self.copy_best))
        actions.addWidget(_button("权重带入检测页", self.send_best_to_predict, "primary"))
        actions.addStretch()
        root.addLayout(actions)

        self.list_widget = QListWidget()
        self.list_widget.currentRowChanged.connect(self.show_detail)
        root.addWidget(self.list_widget, 1)

        self.detail = QPlainTextEdit()
        self.detail.setReadOnly(True)
        self.detail.setMinimumHeight(120)
        root.addWidget(self.detail)

        self.curve_preview = QLabel("暂无训练曲线")
        self.curve_preview.setAlignment(Qt.AlignCenter)
        self.curve_preview.setMinimumHeight(200)
        self.curve_preview.setObjectName("curvePreview")
        root.addWidget(self.curve_preview)

        self.preview_list = ImagePreviewList("暂无结果缩略图", icon_size=128)
        self.preview_list.itemDoubleClicked.connect(self._open_preview_item)
        root.addWidget(self.preview_list)

    def refresh(self):
        self.runs = RunsManager(self.runs_root).list_runs()
        self.list_widget.clear()
        for item in self.runs:
            self.list_widget.addItem(f"[{self._display_type(item)}] {item['modified']}  {item['path']}")
        self.detail.setPlainText(f"共发现 {len(self.runs)} 个结果。")
        self.curve_preview.setPixmap(QPixmap())
        self.curve_preview.setText("暂无训练曲线")
        self.preview_list.set_images([])
        if self.runs:
            self.list_widget.setCurrentRow(0)

    def show_detail(self, index):
        if index < 0 or index >= len(self.runs):
            return
        item = self.runs[index]
        self.detail.setPlainText(
            "\n".join(
                [
                    f"类型: {self._display_type(item)}",
                    f"目录: {item['path']}",
                    f"best.pt: {item['best'] or '无'}",
                    f"last.pt: {item['last'] or '无'}",
                    f"曲线图: {item.get('curve') or '无'}",
                    f"更新时间: {item['modified']}",
                ]
            )
        )
        self._set_curve_preview(item.get("curve"))
        self.preview_list.set_images(item.get("preview_images", []))

    def open_selected(self):
        index = self.list_widget.currentRow()
        if 0 <= index < len(self.runs):
            self._open_path(self.runs[index]["path"])

    def copy_best(self):
        index = self.list_widget.currentRow()
        if 0 <= index < len(self.runs) and self.runs[index]["best"]:
            from PyQt5.QtWidgets import QApplication

            QApplication.clipboard().setText(self.runs[index]["best"])
            QMessageBox.information(self, "完成", "best.pt 路径已复制。")

    def send_best_to_predict(self):
        index = self.list_widget.currentRow()
        if not (0 <= index < len(self.runs)):
            return
        best_path = self.runs[index].get("best", "")
        if not best_path:
            QMessageBox.warning(self, "提示", "当前结果没有 best.pt。")
            return
        home = self.window()
        predict_panel = getattr(home, "panels", [None] * 5)[4] if hasattr(home, "panels") and len(home.panels) > 4 else None
        if predict_panel and hasattr(predict_panel, "set_model_path"):
            predict_panel.set_model_path(best_path)
            if hasattr(home, "switch_panel"):
                home.switch_panel(4)
            QMessageBox.information(self, "完成", "已将权重带入检测页。")

    def _set_curve_preview(self, curve_path):
        if not curve_path or not Path(curve_path).exists():
            self.curve_preview.setPixmap(QPixmap())
            self.curve_preview.setText("暂无训练曲线")
            return
        pixmap = QPixmap(str(curve_path))
        if pixmap.isNull():
            self.curve_preview.setPixmap(QPixmap())
            self.curve_preview.setText("训练曲线读取失败")
            return
        scaled = pixmap.scaled(680, 220, Qt.KeepAspectRatio, Qt.SmoothTransformation)
        self.curve_preview.setPixmap(scaled)
        self.curve_preview.setText("")

    def _open_preview_item(self, item):
        path_text = item.data(Qt.UserRole)
        if path_text:
            self._open_path(path_text)

    def _display_type(self, item):
        lowered = item["path"].lower()
        if "predict" in lowered:
            return "检测"
        if "train" in lowered:
            return "训练"
        return "结果"

_LegacyPanelHome = YOLOHomePanel
_LegacyPanelData = YOLODataPanel
_LegacyPanelTraining = YOLOTrainingPanel
_LegacyPanelPredict = YOLOPredictPanel
_LegacyPanelRuns = YOLORunsPanel
_LegacyPanelSettings = YOLOSettingsPanel


class YOLOProcessPanel(YOLOProcessPanel):
    def refresh_env_display(self):
        if hasattr(self, "env_edit"):
            self.env_edit.setText(self._selected_env_label())


class YOLOHomePanel(_LegacyPanelHome):
    def __init__(self, parent=None):
        self.settings = QSettings("DataForge", "YOLOStudio")
        super().__init__(parent)

    def _build_ui(self):
        root = QVBoxLayout(self)
        root.setContentsMargins(6, 6, 6, 6)
        root.setSpacing(12)

        banner = QLabel("YOLO Studio 将数据准备、环境检测、训练、检测与结果管理集中到一条工作流。")
        banner.setObjectName("banner")
        banner.setWordWrap(True)
        root.addWidget(banner)

        self.metrics = {}
        metrics = QGridLayout()
        metrics.setHorizontalSpacing(10)
        metrics.setVerticalSpacing(10)
        for index, key in enumerate(["环境状态", "当前环境", "最近训练", "最近检测"]):
            card, label = self._metric_card(key, "-")
            self.metrics[key] = label
            metrics.addWidget(card, index // 2, index % 2)
        root.addLayout(metrics)

        action_group = QGroupBox("快捷入口")
        actions = QHBoxLayout(action_group)
        actions.setContentsMargins(12, 16, 12, 12)
        actions.setSpacing(8)
        actions.addWidget(_button("刷新总览", self.refresh, "success"))
        actions.addWidget(_button("打开输出目录", lambda: self._open_path(str(self._default_output_root()))))
        actions.addStretch()
        root.addWidget(action_group)

        self.summary = QPlainTextEdit()
        self.summary.setReadOnly(True)
        self.summary.setMinimumHeight(160)
        root.addWidget(self.summary)
        root.addStretch()

    def refresh(self):
        selected_python = self.settings.value("selected_python", "")
        selected_env_name = self.settings.value("selected_env_name", "当前Python")
        report = EnvironmentChecker.check(selected_python, selected_env_name)
        runs_root = self._default_output_root()
        runs = RunsManager(runs_root).list_runs()
        train_runs = [item for item in runs if "train" in item["path"].lower()]
        predict_runs = [item for item in runs if "predict" in item["path"].lower()]
        weights = [item for item in runs if item.get("best") or item.get("last")]

        self.metrics["环境状态"].setText(report.status)
        self.metrics["当前环境"].setText(selected_env_name)
        self.metrics["最近训练"].setText(train_runs[0]["name"] if train_runs else "暂无")
        self.metrics["最近检测"].setText(predict_runs[0]["name"] if predict_runs else "暂无")

        self.summary.setPlainText(
            "\n".join(
                [
                    f"默认输出目录: {runs_root}",
                    f"训练/检测环境: {selected_env_name}",
                    f"Python: {report.python_version}",
                    f"PyTorch: {report.torch_version}",
                    f"CUDA: {report.cuda_version or '未检测到'}",
                    f"GPU: {report.gpu_name or '无'}",
                    f"可用权重: {len(weights)}",
                    f"Ultralytics: {report.ultralytics_version or '未安装'}",
                ]
            )
        )


class YOLODataPanel(_LegacyPanelData):
    def _build_ui(self):
        super()._build_ui()
        if hasattr(self, "tabs") and self.tabs.count() > 0:
            self.tabs.setTabText(0, "数据集准备")


class YOLOTrainingPanel(_LegacyPanelTraining):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.refresh_output_root()
        self.refresh_env_display()

    def refresh_output_root(self):
        self.project_edit.setText(str(self._default_output_root() / "train"))

    def refresh_env_display(self):
        if hasattr(self, "env_edit"):
            self.env_edit.setText(self._selected_env_label())


class YOLOPredictPanel(_LegacyPanelPredict):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.refresh_output_root()
        self.refresh_env_display()

    def refresh_output_root(self):
        self.output_edit.setText(str(self._default_output_root() / "predict"))

    def refresh_env_display(self):
        if hasattr(self, "env_edit"):
            self.env_edit.setText(self._selected_env_label())


class YOLORunsPanel(_LegacyPanelRuns):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.refresh_output_root()

    def refresh_output_root(self):
        self.runs_root = self._default_output_root()
        self.refresh()
        home = self.window()
        if hasattr(home, "panels"):
            for panel in home.panels:
                if hasattr(panel, "refresh_env_display"):
                    panel.refresh_env_display()
                if hasattr(panel, "refresh") and panel.__class__.__name__ == "YOLOHomePanel":
                    panel.refresh()


class YOLOSettingsPanel(YOLOBasePanel):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.settings = QSettings("DataForge", "YOLOStudio")
        self._build_ui()

    def _build_ui(self):
        root = QVBoxLayout(self)
        root.setContentsMargins(6, 6, 6, 6)
        root.setSpacing(12)

        banner = QLabel("在这里修改 YOLO Studio 的默认输出目录，训练、检测和结果管理会自动跟随。")
        banner.setObjectName("banner")
        banner.setWordWrap(True)
        root.addWidget(banner)

        group = QGroupBox("默认输出")
        layout = QVBoxLayout(group)
        layout.setContentsMargins(12, 16, 12, 12)
        layout.setSpacing(8)

        self.output_root_edit = QLineEdit(str(self._default_output_root()))
        layout.addLayout(_path_row("输出目录", self.output_root_edit, _button("选择", self.choose_output_root)))

        actions = QHBoxLayout()
        actions.addWidget(_button("保存输出目录", self.save_output_root, "success"))
        actions.addWidget(_button("打开目录", lambda: self._open_path(self.output_root_edit.text())))
        actions.addStretch()
        layout.addLayout(actions)
        root.addWidget(group)

        info = QPlainTextEdit()
        info.setReadOnly(True)
        info.setPlainText(
            "说明:\n"
            "1. 这里保存的是软件统一的默认输出根目录。\n"
            "2. 训练默认落到 runs/train。\n"
            "3. 检测默认落到 runs/predict。\n"
            "4. 保存后首页、训练页、检测页和结果管理会同步刷新。"
        )
        root.addWidget(info)
        root.addStretch()

    def choose_output_root(self):
        path = QFileDialog.getExistingDirectory(self, "选择默认输出目录", str(self._default_output_root()))
        if path:
            self.output_root_edit.setText(path)

    def save_output_root(self):
        path_text = self.output_root_edit.text().strip()
        if not path_text:
            QMessageBox.warning(self, "提示", "请输入有效的输出目录。")
            return

        output_root = Path(path_text)
        output_root.mkdir(parents=True, exist_ok=True)
        self.settings.setValue("default_output_dir", str(output_root))

        home = self.window()
        if hasattr(home, "panels"):
            for panel in home.panels:
                if hasattr(panel, "refresh_output_root"):
                    panel.refresh_output_root()
                if hasattr(panel, "refresh_env_display"):
                    panel.refresh_env_display()
                if hasattr(panel, "refresh") and panel.__class__.__name__ == "YOLOHomePanel":
                    panel.refresh()

        QMessageBox.information(self, "完成", "默认输出目录已更新，并已同步到训练、检测和结果页面。")

    def apply_theme(self):
        self.setStyleSheet(
            """
            QLabel#banner {
                background-color: #edf5ff;
                border: 1px solid #c8dcf8;
                border-radius: 14px;
                padding: 12px 16px;
                color: #20456d;
            }

            QGroupBox {
                margin-top: 16px;
                padding-top: 18px;
            }

            QLineEdit {
                border: 1px solid #dbe7f6;
                border-radius: 12px;
                background-color: #fbfdff;
                padding: 10px 12px;
            }
            """
        )


def _perf_load_home_snapshot(selected_python: str, selected_env_name: str, runs_root: str, **_kwargs):
    report = EnvironmentChecker.check(selected_python, selected_env_name)
    runs = RunsManager(Path(runs_root)).list_runs()
    train_runs = [item for item in runs if "train" in item["path"].lower()]
    predict_runs = [item for item in runs if "predict" in item["path"].lower()]
    weights = [item for item in runs if item.get("best") or item.get("last")]
    return {
        "status": report.status,
        "env_name": selected_env_name,
        "train_name": train_runs[0]["name"] if train_runs else "暂无",
        "predict_name": predict_runs[0]["name"] if predict_runs else "暂无",
        "summary": "\n".join(
            [
                f"默认输出目录: {runs_root}",
                f"训练/检测环境: {selected_env_name}",
                f"Python: {report.python_version}",
                f"PyTorch: {report.torch_version}",
                f"CUDA: {report.cuda_version or '未检测到'}",
                f"GPU: {report.gpu_name or '无'}",
                f"可用权重: {len(weights)}",
                f"Ultralytics: {report.ultralytics_version or '未安装'}",
            ]
        ),
    }


def _perf_load_envs(selected_python: str, **_kwargs):
    envs = CondaEnvironmentManager.list_environments()
    selected_index = 0
    for index, env in enumerate(envs):
        if selected_python and env["python"] == selected_python:
            selected_index = index
            break
    return {"envs": envs, "selected_index": selected_index}


def _perf_load_env_report(python_executable: str, env_name: str, saved_python: str, **_kwargs):
    report = EnvironmentChecker.check(python_executable, env_name)
    selected_note = "是" if saved_python and saved_python == report.python_executable else "否"
    return "\n".join(
        [
            f"当前选择: {report.environment_name}",
            f"是否已保存为训练/检测环境: {selected_note}",
            f"Python路径: {report.python_executable}",
            f"状态: {report.status}",
            f"系统: {report.platform}",
            f"Python: {report.python_version}",
            f"PyTorch: {report.torch_version}",
            f"CUDA 可用: {'是' if report.cuda_available else '否'}",
            f"CUDA 版本: {report.cuda_version}",
            f"GPU: {report.gpu_name}",
            f"Ultralytics: {report.ultralytics_version}",
            f"OpenCV: {report.opencv_version}",
            "",
            "CPU 安装命令:",
            report.cpu_install_command,
            "",
            "GPU 安装建议:",
            report.gpu_install_command,
        ]
    )


def _perf_load_runs_snapshot(runs_root: str, **_kwargs):
    return RunsManager(Path(runs_root)).list_runs()


_PerformanceLegacyHome = YOLOHomePanel
_PerformanceLegacyEnvironment = YOLOEnvironmentPanel
_PerformanceLegacyTraining = YOLOTrainingPanel
_PerformanceLegacyPredict = YOLOPredictPanel
_PerformanceLegacyRuns = YOLORunsPanel
_PerformanceLegacySettings = YOLOSettingsPanel


class YOLOHomePanel(_PerformanceLegacyHome):
    def __init__(self, parent=None):
        self._refresh_worker = None
        super().__init__(parent)

    def on_panel_activated(self):
        self.refresh()

    def refresh(self):
        if self._refresh_worker and self._refresh_worker.isRunning():
            return
        self.summary.setPlainText("正在刷新总览，请稍候...")
        selected_python = self.settings.value("selected_python", "")
        selected_env_name = self.settings.value("selected_env_name", "当前Python")
        runs_root = str(self._default_output_root())
        self._refresh_worker = WorkerThread(_perf_load_home_snapshot, selected_python, selected_env_name, runs_root)
        self._refresh_worker.signals.finished.connect(self._apply_perf_snapshot)
        self._refresh_worker.signals.error.connect(self._perf_refresh_failed)
        self._refresh_worker.start()

    def _apply_perf_snapshot(self, snapshot):
        status_key = next((key for key in self.metrics if "状态" in key), list(self.metrics.keys())[0])
        env_key = next((key for key in self.metrics if "环境" in key and "状态" not in key), list(self.metrics.keys())[1])
        train_key = next((key for key in self.metrics if "训练" in key), list(self.metrics.keys())[2])
        predict_key = next((key for key in self.metrics if "检测" in key), list(self.metrics.keys())[3])
        self.metrics[status_key].setText(snapshot["status"])
        self.metrics[env_key].setText(snapshot["env_name"])
        self.metrics[train_key].setText(snapshot["train_name"])
        self.metrics[predict_key].setText(snapshot["predict_name"])
        self.summary.setPlainText(snapshot["summary"])
        self._refresh_worker = None

    def _perf_refresh_failed(self, error_text):
        self.summary.setPlainText(f"总览刷新失败：\n{error_text}")
        self._refresh_worker = None


class YOLOEnvironmentPanel(_PerformanceLegacyEnvironment):
    def __init__(self, parent=None):
        self._envs_worker = None
        self._report_worker = None
        super().__init__(parent)

    def on_panel_activated(self):
        self.refresh_envs()
        self.refresh()

    def refresh_envs(self):
        if self._envs_worker and self._envs_worker.isRunning():
            return
        selected_python = self.settings.value("selected_python", "")
        self.env_combo.clear()
        self.env_combo.addItem("正在加载环境...", None)
        self._envs_worker = WorkerThread(_perf_load_envs, selected_python)
        self._envs_worker.signals.finished.connect(self._apply_perf_envs)
        self._envs_worker.signals.error.connect(self._perf_envs_failed)
        self._envs_worker.start()

    def _apply_perf_envs(self, payload):
        self.conda_envs = payload["envs"]
        self.env_combo.clear()
        for env in self.conda_envs:
            self.env_combo.addItem(f"{env['name']}  |  {env['python']}", env)
        if self.conda_envs:
            self.env_combo.setCurrentIndex(payload["selected_index"])
        self._envs_worker = None

    def _perf_envs_failed(self, error_text):
        self.env_combo.clear()
        self.env_combo.addItem("环境加载失败", None)
        self.report.setPlainText(f"环境列表加载失败：\n{error_text}")
        self._envs_worker = None

    def refresh(self):
        if self._report_worker and self._report_worker.isRunning():
            return
        env = self.env_combo.currentData() if hasattr(self, "env_combo") else None
        python_executable = env["python"] if env else ""
        env_name = env["name"] if env else "当前环境"
        saved_python = self.settings.value("selected_python", "")
        self.report.setPlainText("正在检测当前环境，请稍候...")
        self._report_worker = WorkerThread(_perf_load_env_report, python_executable, env_name, saved_python)
        self._report_worker.signals.finished.connect(self._apply_perf_report)
        self._report_worker.signals.error.connect(self._perf_report_failed)
        self._report_worker.start()

    def _apply_perf_report(self, text):
        self.report.setPlainText(text)
        self._report_worker = None

    def _perf_report_failed(self, error_text):
        self.report.setPlainText(f"环境检测失败：\n{error_text}")
        self._report_worker = None

    def save_selected_env(self):
        env = self.env_combo.currentData()
        if not env:
            QMessageBox.warning(self, "提示", "没有可保存的环境。")
            return
        self.settings.setValue("selected_env_name", env["name"])
        self.settings.setValue("selected_env_path", env["path"])
        self.settings.setValue("selected_python", env["python"])
        QMessageBox.information(self, "完成", f"已切换到环境：{env['name']}")
        self.refresh()
        home = self.window()
        if hasattr(home, "panels"):
            for panel in home.panels:
                if panel is None:
                    continue
                if hasattr(panel, "refresh_env_display"):
                    panel.refresh_env_display()
                if hasattr(panel, "refresh") and panel.__class__.__name__ == "YOLOHomePanel":
                    panel.refresh()


class YOLOTrainingPanel(_PerformanceLegacyTraining):
    def on_panel_activated(self):
        self.refresh_output_root()
        self.refresh_env_display()


class YOLOPredictPanel(_PerformanceLegacyPredict):
    def on_panel_activated(self):
        self.refresh_output_root()
        self.refresh_env_display()


class YOLORunsPanel(_PerformanceLegacyRuns):
    def __init__(self, parent=None):
        self._runs_worker = None
        super().__init__(parent)

    def on_panel_activated(self):
        self.refresh_output_root()
        self.refresh()

    def refresh_output_root(self):
        self.runs_root = self._default_output_root()

    def refresh(self):
        if self._runs_worker and self._runs_worker.isRunning():
            return
        self.list_widget.clear()
        self.detail.setPlainText("正在扫描结果目录，请稍候...")
        self._runs_worker = WorkerThread(_perf_load_runs_snapshot, str(self.runs_root))
        self._runs_worker.signals.finished.connect(self._apply_perf_runs)
        self._runs_worker.signals.error.connect(self._perf_runs_failed)
        self._runs_worker.start()

    def _apply_perf_runs(self, runs):
        self.runs = runs
        self.list_widget.clear()
        for item in self.runs:
            self.list_widget.addItem(f"[{self._display_type(item)}] {item['modified']}  {item['path']}")
        self.detail.setPlainText(f"共发现 {len(self.runs)} 个结果。")
        self.curve_preview.setPixmap(QPixmap())
        self.curve_preview.setText("暂无训练曲线")
        self.preview_list.set_images([])
        if self.runs:
            self.list_widget.setCurrentRow(0)
        self._runs_worker = None

    def _perf_runs_failed(self, error_text):
        self.list_widget.clear()
        self.detail.setPlainText(f"结果扫描失败：\n{error_text}")
        self._runs_worker = None


class YOLOSettingsPanel(_PerformanceLegacySettings):
    def save_output_root(self):
        path_text = self.output_root_edit.text().strip()
        if not path_text:
            QMessageBox.warning(self, "提示", "请输入有效的输出目录。")
            return

        output_root = Path(path_text)
        output_root.mkdir(parents=True, exist_ok=True)
        self.settings.setValue("default_output_dir", str(output_root))

        home = self.window()
        if hasattr(home, "panels"):
            for panel in home.panels:
                if panel is None:
                    continue
                if hasattr(panel, "refresh_output_root"):
                    panel.refresh_output_root()
                if hasattr(panel, "refresh_env_display"):
                    panel.refresh_env_display()
                if hasattr(panel, "refresh") and panel.__class__.__name__ == "YOLOHomePanel":
                    panel.refresh()

        QMessageBox.information(self, "完成", "默认输出目录已更新，并已同步到训练、检测和结果页面。")
