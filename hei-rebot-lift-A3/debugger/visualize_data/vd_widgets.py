# -*- coding: utf-8 -*-
"""通用 UI 控件：可拖动剪辑滑条、RECAP 奖励标注轨道、等比例缩放图像标签、全分辨率图像弹窗。"""
import numpy as np
import cv2
from PyQt6.QtWidgets import QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton, QScrollArea, QSizePolicy, QDialog
from PyQt6.QtCore import Qt, pyqtSignal, QPointF
from PyQt6.QtGui import QImage, QPixmap, QPainter, QBrush, QPen, QColor, QPolygonF


# ==================== 可拖动剪辑标记滑条 ====================
class TrimSlider(QWidget):
    """自定义三标记滑条：起止标记 + 可拖动播放头。

    双击滑条时发出 segment_selected 信号，用于将位置曲线缩放到所选片段。
    """
    markers_changed = pyqtSignal(int, int)
    position_changed = pyqtSignal(int)  # 播放头位置实时变化（拖动/点击）
    position_clicked = pyqtSignal(int)    # 点击轨道/释放播放头时触发
    segment_selected = pyqtSignal(int, int)  # 双击选中起止片段

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumHeight(58)
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self._total = 0
        self._current = 0
        self._start = 0
        self._end = 0
        self._dragging = None  # 'start' / 'end' / 'playhead' / None

    def mouseDoubleClickEvent(self, event):
        """双击选中当前起止标记之间的片段。"""
        if self._total <= 0:
            return
        start, end = self.get_range()
        self.segment_selected.emit(start, end)
        super().mouseDoubleClickEvent(event)

    def set_total_frames(self, total):
        self._total = max(total, 0)
        self._start = 0
        self._end = max(0, self._total - 1)
        self._current = 0
        self.update()
        if self._total > 0:
            self.markers_changed.emit(self._start, self._end)

    def set_current_frame(self, frame):
        self._current = max(0, min(frame, max(0, self._total - 1)))
        self.update()

    def set_range(self, start, end):
        max_frame = max(0, self._total - 1)
        self._start = max(0, min(start, max_frame))
        self._end = max(0, min(end, max_frame))
        self.update()
        self.markers_changed.emit(self._start, self._end)

    def get_range(self):
        return (min(self._start, self._end), max(self._start, self._end))

    def _frame_from_x(self, x):
        rect = self.rect()
        if rect.width() <= 0 or self._total <= 0:
            return 0
        margin = 12
        w = rect.width() - 2 * margin
        ratio = max(0.0, min(1.0, (x - margin) / w))
        # 分母用 total-1：帧 0 在左端、帧 total-1 在右端（修复右端 off-by-one 越界）
        return int(round(ratio * max(1, self._total - 1)))

    def _x_from_frame(self, frame):
        rect = self.rect()
        if self._total <= 0:
            return rect.left() + 12
        margin = 12
        w = rect.width() - 2 * margin
        # 与 _frame_from_x 同一映射：分母 total-1
        ratio = max(0.0, min(1.0, frame / max(1, self._total - 1)))
        return rect.left() + margin + int(ratio * w)

    def mousePressEvent(self, event):
        if self._total <= 0:
            return
        x = int(event.position().x())
        start_x = self._x_from_frame(self._start)
        end_x = self._x_from_frame(self._end)
        cur_x = self._x_from_frame(self._current)
        if abs(x - start_x) < 10:
            self._dragging = 'start'
        elif abs(x - end_x) < 10:
            self._dragging = 'end'
        elif abs(x - cur_x) < 12:
            self._dragging = 'playhead'
        else:
            # 点击轨道跳转当前位置
            self._dragging = None
            frame = self._frame_from_x(x)
            self._current = frame
            self.update()
            self.position_clicked.emit(frame)

    def mouseMoveEvent(self, event):
        if self._dragging is None or self._total <= 0:
            return
        frame = self._frame_from_x(int(event.position().x()))
        if self._dragging == 'start':
            self._start = max(0, min(frame, self._total - 1))
            self.update()
            self.markers_changed.emit(self._start, self._end)
        elif self._dragging == 'end':
            self._end = max(0, min(frame, self._total - 1))
            self.update()
            self.markers_changed.emit(self._start, self._end)
        elif self._dragging == 'playhead':
            self._current = max(0, min(frame, self._total - 1))
            self.update()
            self.position_changed.emit(self._current)

    def mouseReleaseEvent(self, event):
        if self._dragging == 'playhead':
            self.position_clicked.emit(self._current)
        self._dragging = None

    def leaveEvent(self, event):
        self._dragging = None

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        rect = self.rect()
        margin = 12
        groove = rect.adjusted(margin, 10, -margin, -18)

        # 背景 groove
        painter.setBrush(QBrush(QColor("#555555")))
        painter.setPen(Qt.PenStyle.NoPen)
        painter.drawRoundedRect(groove, 4, 4)

        if self._total <= 0:
            painter.end()
            return

        # 选中区域（两个标记之间）
        start_x = self._x_from_frame(self._start)
        end_x = self._x_from_frame(self._end)
        sel_rect = groove.adjusted(0, 0, 0, 0)
        sel_rect.setLeft(min(start_x, end_x))
        sel_rect.setRight(max(start_x, end_x))
        painter.setBrush(QBrush(QColor("#FF5722")))
        painter.drawRoundedRect(sel_rect, 4, 4)

        # 播放头（可拖动）
        cur_x = self._x_from_frame(self._current)
        ph_top = groove.top() - 10
        ph_bottom = groove.bottom() + 10
        painter.setPen(QPen(QColor("#4CAF50"), 2))
        painter.drawLine(cur_x, ph_top, cur_x, ph_bottom)
        # 播放头小三角
        painter.setBrush(QBrush(QColor("#4CAF50")))
        painter.setPen(Qt.PenStyle.NoPen)
        ph_tri = QPolygonF([
            QPointF(cur_x, ph_top - 2),
            QPointF(cur_x - 6, ph_top - 12),
            QPointF(cur_x + 6, ph_top - 12),
        ])
        painter.drawPolygon(ph_tri)

        # 拖动标记三角形 + 文字（起止剪辑标记）
        painter.setPen(QPen(QColor("#333333"), 1))
        for fx, kind in [(start_x, 'start'), (end_x, 'end')]:
            color = QColor("#2196F3") if kind == 'start' else QColor("#9C27B0")
            painter.setBrush(QBrush(color))
            tri = QPolygonF([
                QPointF(fx, groove.bottom() + 2),
                QPointF(fx - 7, groove.bottom() + 16),
                QPointF(fx + 7, groove.bottom() + 16),
            ])
            painter.drawPolygon(tri)
            # 标记文字
            frame_val = self._start if kind == 'start' else self._end
            text = str(frame_val)
            painter.setPen(QPen(QColor("#FFFFFF")))
            painter.drawText(int(fx) - 12, groove.bottom() + 30, 24, 14, Qt.AlignmentFlag.AlignCenter, text)
        # 播放头文字
        painter.setPen(QPen(QColor("#4CAF50")))
        painter.drawText(int(cur_x) - 16, ph_top - 18, 32, 14, Qt.AlignmentFlag.AlignCenter, str(self._current))
        painter.end()


# ==================== RECAP 奖励标注轨道 ====================
class RecapTrackWidget(QWidget):
    """RECAP 奖励标注轨道：在播放时间线下方用彩色细带显示标注区间。

    灰色=未标记，绿色=good，红色=bad，橙色=neutral，蓝色=干预段（人工接管
    纠正，is_human_intervention=True，覆盖标签色）。宽度和边距与 TrimSlider 对齐。
    双击某个色条发出 interval_double_clicked（区间下标），供主窗口进入编辑模式。
    """

    LABEL_COLORS = {"good": "#2ecc71", "bad": "#e74c3c", "neutral": "#f39c12"}
    INTERVENTION_COLOR = "#3498db"

    interval_double_clicked = pyqtSignal(int)  # 双击色条：参数为区间下标

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumHeight(12)
        self.setMaximumHeight(12)
        self._total = 0
        self._current = 0
        self._intervals = []

    def set_total(self, total):
        self._total = max(0, total)
        self.update()

    def set_intervals(self, intervals):
        self._intervals = list(intervals or [])
        self.update()

    def set_current(self, frame):
        self._current = max(0, frame)
        self.update()

    def _x_from_frame(self, frame):
        """与 TrimSlider 相同的坐标映射（margin=12，分母 total-1）。"""
        if self._total <= 0:
            return 12
        margin = 12
        w = self.width() - 2 * margin
        ratio = max(0.0, min(1.0, frame / max(1, self._total - 1)))
        return margin + int(ratio * w)

    def mouseDoubleClickEvent(self, event):
        """双击色条：定位所在（或最近的）区间并发出编辑请求。"""
        if self._total > 0 and self._intervals:
            margin = 12
            w = self.width() - 2 * margin
            if w > 0:
                ratio = max(0.0, min(1.0, (int(event.position().x()) - margin) / w))
                frame = ratio * max(1, self._total - 1)
                best, best_dist = None, None
                for i, iv in enumerate(self._intervals):
                    if iv["start"] <= frame <= iv["end"] + 1:
                        best = i
                        break
                    d = min(abs(frame - iv["start"]), abs(frame - iv["end"]))
                    if best is None or d < best_dist:
                        best, best_dist = i, d
                if best is not None:
                    self.interval_double_clicked.emit(best)
        super().mouseDoubleClickEvent(event)

    def paintEvent(self, event):
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        w = self.width()
        h = self.height()
        margin = 12
        track_w = w - 2 * margin
        track_y = 1
        track_h = h - 2

        if track_w <= 0:
            return

        if self._total <= 0:
            p.setBrush(QColor("#3a3a3a"))
            p.setPen(Qt.PenStyle.NoPen)
            p.drawRoundedRect(margin, track_y, track_w, track_h, 2, 2)
            return

        # 灰色底（未标记区域）
        p.setBrush(QColor("#888888"))
        p.setPen(Qt.PenStyle.NoPen)
        p.drawRoundedRect(margin, track_y, track_w, track_h, 2, 2)

        # 彩色区间段（覆盖在灰色底之上）；干预段（人工接管纠正）显示为蓝色
        for iv in self._intervals:
            x1 = self._x_from_frame(iv["start"])
            x2 = self._x_from_frame(iv["end"] + 1)
            if iv.get("is_human_intervention"):
                color = QColor(self.INTERVENTION_COLOR)
            else:
                color = QColor(self.LABEL_COLORS.get(iv.get("label"), "#888888"))
            p.setBrush(color)
            p.drawRoundedRect(x1, track_y, max(2, x2 - x1), track_h, 2, 2)

        # 当前播放位置线
        cx = margin + int((self._current / max(1, self._total - 1)) * track_w)
        p.setPen(QPen(QColor("#ffffff"), 1))
        p.drawLine(cx, 0, cx, h)


# ==================== 等比例缩放图像标签 ====================
class ImageLabel(QLabel):
    """固定 16:9（128:72）显示区域，保持原图比例缩放，避免画面被压缩或拉伸。
    
    支持鼠标双击事件，用于弹出全分辨率图像。
    """
    
    ASPECT_W = 128
    ASPECT_H = 72
    doubleClicked = pyqtSignal()
    
    def __init__(self, text="", parent=None):
        super().__init__(text, parent)
        self._original = None       # 原始 numpy 图像 (H, W, C)
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        # 固定 16:9 显示区域，高度根据宽度自动计算
        self.setSizePolicy(
            QSizePolicy.Policy.Expanding,
            QSizePolicy.Policy.Preferred,
        )
    
    def mouseDoubleClickEvent(self, event):
        """双击时发出信号，通知主窗口弹出全分辨率图像。"""
        self.doubleClicked.emit()
        super().mouseDoubleClickEvent(event)
    
    def setImage(self, image):
        """设置原始图像，可传入 numpy 数组 (H,W,C) 或 QPixmap"""
        if isinstance(image, QPixmap):
            if image.isNull():
                self._original = None
            else:
                qimg = image.toImage().convertToFormat(QImage.Format.Format_RGB888)
                w, h = qimg.width(), qimg.height()
                ptr = qimg.bits()
                if hasattr(ptr, 'asarray'):
                    arr = np.array(ptr.asarray(h * w * 3)).reshape((h, w, 3))
                else:
                    arr = np.array(ptr).reshape((h, w, 3))
                self._original = arr
        elif isinstance(image, np.ndarray):
            self._original = image.copy() if image is not None else None
        else:
            self._original = None
        
        self.updateGeometry()
        self._apply_scaling()
    
    def setPixmap(self, pixmap):
        """兼容 QLabel.setPixmap"""
        self.setImage(pixmap)
    
    def clearImage(self):
        self._original = None
        self.updateGeometry()
        super().setPixmap(QPixmap())
    
    def hasHeightForWidth(self):
        return True
    
    def heightForWidth(self, width):
        """固定 16:9 比例"""
        return int(width * ImageLabel.ASPECT_H / ImageLabel.ASPECT_W)
    
    def sizeHint(self):
        from PyQt6.QtCore import QSize
        return QSize(self.ASPECT_W * 3, self.ASPECT_H * 3)  # 384 x 216 默认
    
    def resizeEvent(self, event):
        super().resizeEvent(event)
        self._apply_scaling()
    
    def _apply_scaling(self):
        if self._original is None:
            super().setPixmap(QPixmap())
            return
        
        label_w = max(self.width(), 1)
        label_h = max(self.height(), 1)
        img_h, img_w = self._original.shape[:2]
        
        # 在 16:9 的显示框内，按原图比例等比例缩放（不拉伸），可能会有黑边
        scale = min(label_w / img_w, label_h / img_h)
        if scale > 1.0:
            new_w, new_h = img_w, img_h
        else:
            new_w, new_h = int(img_w * scale), int(img_h * scale)
        
        if new_w <= 0 or new_h <= 0:
            super().setPixmap(QPixmap())
            return
        
        try:
            import cv2
            resized = cv2.resize(self._original, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
            h, w, ch = resized.shape
            bytes_per_line = ch * w
            q_image = QImage(resized.data, w, h, bytes_per_line, QImage.Format.Format_RGB888)
            pixmap = QPixmap.fromImage(q_image)
        except Exception:
            pixmap = QPixmap.fromImage(
                QImage(self._original.data, img_w, img_h, img_w * 3, QImage.Format.Format_RGB888)
            ).scaled(new_w, new_h, Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation)
        
        super().setPixmap(pixmap)


# ==================== 全分辨率图像弹窗 ====================
class FullscreenImageDialog(QDialog):
    """双击图像回放框后弹出的全分辨率图像查看窗口。

    使用 QScrollArea 承载原始分辨率图像，支持滚动查看细节。
    """

    def __init__(self, parent, image, title="全分辨率图像"):
        super().__init__(parent)
        self.setWindowTitle(title)
        self.resize(1200, 800)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(10, 10, 10, 10)

        # 滚动区域，方便查看大分辨率图像
        self.scroll_area = QScrollArea(self)
        self.scroll_area.setWidgetResizable(True)
        self.scroll_area.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(self.scroll_area)

        self.image_label = QLabel(self)
        self.image_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.scroll_area.setWidget(self.image_label)

        # 将 numpy 图像转换为 QPixmap 并显示
        if isinstance(image, np.ndarray) and image.size > 0:
            rgb = np.ascontiguousarray(image[..., :3])
            h, w, ch = rgb.shape
            q_image = QImage(rgb.data, w, h, ch * w, QImage.Format.Format_RGB888)
            pixmap = QPixmap.fromImage(q_image.copy())
            self.image_label.setPixmap(pixmap)
            self.image_label.setFixedSize(w, h)

        # 关闭按钮
        btn_layout = QHBoxLayout()
        btn_layout.addStretch()
        close_btn = QPushButton("关闭")
        close_btn.setStyleSheet("QPushButton { padding: 6px 20px; }")
        close_btn.clicked.connect(self.accept)
        btn_layout.addWidget(close_btn)
        layout.addLayout(btn_layout)
