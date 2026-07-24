"""
Product-style blue workspace shell for DataForge YOLO Studio.
"""

from pathlib import Path

from PyQt5.QtCore import Qt, pyqtSignal
from PyQt5.QtGui import QCursor, QIcon
from PyQt5.QtWidgets import (
    QFrame,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QScrollArea,
    QStackedWidget,
    QVBoxLayout,
    QWidget,
)

from .styles import AppStyles
from .theme_manager import theme_manager


class NavCard(QFrame):
    """Compact product navigation item."""

    clicked = pyqtSignal(int)

    def __init__(self, index: int, title: str, subtitle: str, parent=None):
        super().__init__(parent)
        self.index = index
        self.setObjectName("navCard")
        self.setProperty("selected", "false")
        self.setCursor(QCursor(Qt.PointingHandCursor))
        self.setFixedHeight(44)

        layout = QHBoxLayout(self)
        layout.setContentsMargins(8, 4, 10, 4)
        layout.setSpacing(8)

        self.index_label = QLabel(f"{index + 1:02d}")
        self.index_label.setObjectName("navIndex")
        self.index_label.setAlignment(Qt.AlignCenter)
        self.index_label.setFixedSize(24, 24)

        self.title_label = QLabel(title)
        self.title_label.setObjectName("navTitle")
        layout.addWidget(self.index_label)
        layout.addWidget(self.title_label, 1)

    def set_selected(self, selected: bool):
        self.setProperty("selected", "true" if selected else "false")
        self.style().unpolish(self)
        self.style().polish(self)

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            self.clicked.emit(self.index)
        super().mousePressEvent(event)


class SimpleHomeWindow(QMainWindow):
    """White workspace card with a compact blue product navigation."""

    NAV_ITEMS = [
        ("总览", "环境与任务概览"),
        ("数据准备", "检查、转换与划分"),
        ("环境检测", "Python、Conda 与 GPU"),
        ("模型训练", "配置并启动训练"),
        ("模型检测", "权重推理与预览"),
        ("结果管理", "曲线、结果与权重"),
        ("设置", "输出目录与偏好"),
    ]

    def __init__(self):
        super().__init__()
        theme_manager.set_theme("light")
        self._sync_app_styles()
        self.nav_cards = []
        self.panels = []
        self.panel_classes = []
        self.panel_placeholders = []
        self.current_index = 0
        self._build_ui()
        self._build_panels()
        self._apply_shell_style()
        self.switch_panel(0)

    def _sync_app_styles(self):
        colors = theme_manager.get_theme_config("light")["colors"]
        AppStyles.PRIMARY_COLOR = colors["primary"]
        AppStyles.SUCCESS_COLOR = colors["success"]
        AppStyles.WARNING_COLOR = colors["warning"]
        AppStyles.DANGER_COLOR = colors["danger"]
        AppStyles.BACKGROUND_COLOR = colors["background"]
        AppStyles.CARD_COLOR = colors["card"]
        AppStyles.TEXT_COLOR = colors["text"]
        AppStyles.SECONDARY_TEXT = colors["secondary_text"]
        AppStyles.BORDER_COLOR = colors["border"]
        AppStyles.SIDEBAR_BG = colors["sidebar_bg"]
        AppStyles.SIDEBAR_TEXT = colors["sidebar_text"]
        AppStyles.SIDEBAR_HOVER = colors["sidebar_hover"]
        AppStyles.NAV_SELECTED_BG = colors["nav_selected"]
        AppStyles.NAV_SELECTED_BORDER = colors["nav_selected_border"]
        AppStyles.HEADER_BG = colors["header_bg"]
        AppStyles.HEADER_TEXT = colors["header_text"]

    def _icon_path(self) -> Path:
        icon_root = Path(__file__).resolve().parents[3] / "assets"
        for name in ("logo.ico", "logo.png"):
            candidate = icon_root / name
            if candidate.exists():
                return candidate
        return Path(__file__).parent.parent.parent / "resources" / "icon.png"

    def _build_ui(self):
        self.setWindowTitle("DataForge YOLO Studio")
        self.setMinimumSize(760, 520)
        self.resize(980, 620)

        icon_path = self._icon_path()
        if icon_path.exists():
            self.setWindowIcon(QIcon(str(icon_path)))

        root = QWidget()
        root.setObjectName("workspaceRoot")
        self.setCentralWidget(root)

        root_layout = QVBoxLayout(root)
        root_layout.setContentsMargins(8, 8, 8, 8)
        root_layout.setSpacing(0)

        self.workspace_card = QFrame()
        self.workspace_card.setObjectName("workspaceCard")
        shell_layout = QHBoxLayout(self.workspace_card)
        shell_layout.setContentsMargins(0, 0, 0, 0)
        shell_layout.setSpacing(0)
        root_layout.addWidget(self.workspace_card, 1)

        self.sidebar = QFrame()
        self.sidebar.setObjectName("sidebar")
        self.sidebar.setFixedWidth(190)
        sidebar_layout = QVBoxLayout(self.sidebar)
        sidebar_layout.setContentsMargins(8, 8, 8, 8)
        sidebar_layout.setSpacing(4)

        nav_container = QWidget()
        nav_container.setObjectName("navContainer")
        self.nav_layout = QVBoxLayout(nav_container)
        self.nav_layout.setContentsMargins(0, 0, 0, 0)
        self.nav_layout.setSpacing(0)

        for index, (title, subtitle) in enumerate(self.NAV_ITEMS):
            card = NavCard(index, title, subtitle, self)
            card.clicked.connect(self.switch_panel)
            self.nav_cards.append(card)
            self.nav_layout.addWidget(card)
        self.nav_layout.addStretch()

        self.nav_scroll = QScrollArea()
        self.nav_scroll.setObjectName("navScroll")
        self.nav_scroll.setWidgetResizable(True)
        self.nav_scroll.setFrameShape(QFrame.NoFrame)
        self.nav_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.nav_scroll.setWidget(nav_container)
        sidebar_layout.addWidget(self.nav_scroll, 1)

        footer = QLabel("LOCAL WORKSPACE\n本机训练与检测")
        footer.setObjectName("sidebarFooter")
        sidebar_layout.addWidget(footer)
        footer.hide()
        shell_layout.addWidget(self.sidebar)

        self.content_wrap = QFrame()
        self.content_wrap.setObjectName("contentWrap")
        content_layout = QVBoxLayout(self.content_wrap)
        content_layout.setContentsMargins(14, 12, 14, 12)
        content_layout.setSpacing(10)
        shell_layout.addWidget(self.content_wrap, 1)

        self.header_card = QFrame()
        self.header_card.setObjectName("headerCard")
        header_layout = QHBoxLayout(self.header_card)
        header_layout.setContentsMargins(2, 0, 2, 0)
        header_layout.setSpacing(12)

        title_layout = QVBoxLayout()
        title_layout.setContentsMargins(0, 0, 0, 0)
        title_layout.setSpacing(3)
        self.page_title = QLabel("总览")
        self.page_title.setObjectName("pageTitle")
        self.page_desc = QLabel("查看环境、训练、检测与输出结果的整体状态")
        self.page_desc.setObjectName("pageDesc")
        title_layout.addWidget(self.page_title)
        title_layout.addWidget(self.page_desc)
        header_layout.addLayout(title_layout, 1)

        self.workspace_badge = QLabel("本地工作区")
        self.workspace_badge.setObjectName("workspaceBadge")
        self.workspace_badge.setAlignment(Qt.AlignCenter)
        header_layout.addWidget(self.workspace_badge, 0, Qt.AlignTop)
        content_layout.addWidget(self.header_card)

        self.stack_card = QFrame()
        self.stack_card.setObjectName("stackCard")
        stack_layout = QVBoxLayout(self.stack_card)
        stack_layout.setContentsMargins(8, 8, 8, 8)
        stack_layout.setSpacing(0)

        self.stack_scroll = QScrollArea()
        self.stack_scroll.setObjectName("contentScroll")
        self.stack_scroll.setWidgetResizable(True)
        self.stack_scroll.setFrameShape(QFrame.NoFrame)
        self.stack_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)

        self.stack = QStackedWidget()
        self.stack.setObjectName("contentStack")
        self.stack_scroll.setWidget(self.stack)
        stack_layout.addWidget(self.stack_scroll)
        content_layout.addWidget(self.stack_card, 1)

    def _build_panels(self):
        from .yolo_panels import (
            YOLODataPanel,
            YOLOEnvironmentPanel,
            YOLOHomePanel,
            YOLOPredictPanel,
            YOLORunsPanel,
            YOLOSettingsPanel,
            YOLOTrainingPanel,
        )

        self.panel_classes = [
            YOLOHomePanel,
            YOLODataPanel,
            YOLOEnvironmentPanel,
            YOLOTrainingPanel,
            YOLOPredictPanel,
            YOLORunsPanel,
            YOLOSettingsPanel,
        ]
        self.panels = [None] * len(self.panel_classes)
        self.panel_placeholders = []
        for index in range(len(self.panel_classes)):
            placeholder = self._create_panel_placeholder(index)
            self.panel_placeholders.append(placeholder)
            self.stack.addWidget(placeholder)

    def _create_panel_placeholder(self, index: int):
        title, subtitle = self.NAV_ITEMS[index]
        container = QFrame()
        container.setObjectName("panelPlaceholder")
        layout = QVBoxLayout(container)
        layout.setContentsMargins(24, 24, 24, 24)
        layout.setSpacing(8)

        title_label = QLabel(title)
        title_label.setObjectName("placeholderTitle")
        desc_label = QLabel(f"{subtitle}\n页面正在初始化，请稍候。")
        desc_label.setObjectName("placeholderDesc")
        desc_label.setWordWrap(True)
        layout.addStretch()
        layout.addWidget(title_label, 0, Qt.AlignHCenter)
        layout.addWidget(desc_label, 0, Qt.AlignHCenter)
        layout.addStretch()
        return container

    def _ensure_panel(self, index: int):
        panel = self.panels[index]
        if panel is not None:
            return panel

        panel = self.panel_classes[index](self)
        if hasattr(panel, "apply_theme"):
            panel.apply_theme()

        placeholder = self.panel_placeholders[index]
        self.stack.insertWidget(index, panel)
        self.stack.removeWidget(placeholder)
        placeholder.deleteLater()
        self.panels[index] = panel
        self.panel_placeholders[index] = None
        return panel

    def switch_panel(self, index: int):
        if index < 0 or index >= len(self.NAV_ITEMS):
            return

        current_panel = self._ensure_panel(index)
        self.current_index = index
        self.stack.setCurrentIndex(index)

        title, subtitle = self.NAV_ITEMS[index]
        self.page_title.setText(title)
        self.page_desc.setText(subtitle)

        for card_index, card in enumerate(self.nav_cards):
            card.set_selected(card_index == index)

        if hasattr(current_panel, "apply_theme"):
            current_panel.apply_theme()
        if hasattr(current_panel, "refresh_output_root"):
            current_panel.refresh_output_root()
        if hasattr(current_panel, "refresh_env_display"):
            current_panel.refresh_env_display()
        if hasattr(current_panel, "on_panel_activated"):
            current_panel.on_panel_activated()
        elif hasattr(current_panel, "refresh"):
            current_panel.refresh()

    def _apply_shell_style(self):
        shell_style = """
        QWidget#workspaceRoot {
            background-color: #eeeeee;
        }

        QFrame#workspaceCard {
            background-color: #ffffff;
            border: 1px solid #bdbdbd;
            border-radius: 0px;
        }

        QFrame#sidebar {
            background-color: #f3f3f3;
            border: none;
            border-right: 1px solid #c8c8c8;
        }

        QFrame#brandBlock {
            background-color: transparent;
            border: none;
            border-radius: 0px;
        }

        QLabel#brandLogo {
            background-color: transparent;
            border-radius: 0px;
        }

        QLabel#brandTitle {
            color: #202020;
            font-size: 14px;
            font-weight: bold;
            background-color: transparent;
        }

        QLabel#brandSubtitle {
            color: #666666;
            font-size: 10px;
            background-color: transparent;
        }

        QLabel#sectionLabel {
            color: #666666;
            font-size: 9px;
            font-weight: bold;
            padding: 5px 8px 2px 8px;
            background-color: transparent;
        }

        QFrame#navCard {
            background-color: transparent;
            border: none;
            border-bottom: 1px solid #dddddd;
            border-radius: 0px;
        }

        QFrame#navCard:hover {
            background-color: #e6e6e6;
        }

        QFrame#navCard[selected="true"] {
            background-color: #dce8f8;
            border-left: 3px solid #3f6fae;
        }

        QFrame#navCard QLabel#navIndex {
            color: #666666;
            background-color: transparent;
            border-radius: 0px;
            font-size: 9px;
            font-weight: bold;
        }

        QFrame#navCard[selected="true"] QLabel#navIndex {
            color: #315f99;
            background-color: transparent;
        }

        QFrame#navCard QLabel#navTitle {
            color: #222222;
            font-size: 12px;
            font-weight: bold;
            background-color: transparent;
        }

        QFrame#navCard QLabel#navSubtitle {
            color: #777777;
            font-size: 9px;
            background-color: transparent;
        }

        QLabel#sidebarFooter {
            color: #777777;
            font-size: 9px;
            padding: 8px;
            background-color: transparent;
        }

        QFrame#contentWrap {
            background-color: transparent;
        }

        QFrame#headerCard {
            background-color: transparent;
            border: none;
        }

        QLabel#pageTitle {
            color: #222222;
            font-size: 18px;
            font-weight: bold;
            background-color: transparent;
        }

        QLabel#pageDesc {
            color: #666666;
            font-size: 11px;
            background-color: transparent;
        }

        QLabel#workspaceBadge {
            color: #555555;
            background-color: #eeeeee;
            border: 1px solid #cccccc;
            border-radius: 0px;
            padding: 5px 8px;
            font-size: 9px;
            font-weight: bold;
        }

        QFrame#stackCard {
            background-color: #ffffff;
            border: 1px solid #c8c8c8;
            border-radius: 0px;
        }

        QStackedWidget#contentStack,
        QScrollArea#contentScroll,
        QScrollArea#navScroll {
            background-color: transparent;
            border: none;
        }

        QWidget#navContainer,
        QScrollArea#contentScroll > QWidget > QWidget {
            background-color: transparent;
        }

        QFrame#panelPlaceholder {
            background-color: #ffffff;
            border: 1px solid #cccccc;
            border-radius: 0px;
        }

        QLabel#placeholderTitle {
            color: #222222;
            font-size: 16px;
            font-weight: bold;
            background-color: transparent;
        }

        QLabel#placeholderDesc {
            color: #666666;
            font-size: 11px;
            background-color: transparent;
        }
        """
        self.setStyleSheet(theme_manager.generate_stylesheet("light") + shell_style)
