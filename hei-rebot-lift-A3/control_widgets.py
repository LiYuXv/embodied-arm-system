# -*- coding: utf-8 -*-
"""
底盘/升降控制面板的自定义控件：二维摇杆 + 一维水平摇杆。
拖动输出 [-1, 1]，松开自动回中并输出 0。
"""

from PyQt6 import QtCore, QtGui, QtWidgets


class JoystickWidget(QtWidgets.QWidget):
    """二维摇杆：方框内圆点，拖动方向即运动方向，偏离中心越远速度越大；松开回中。"""

    valueChanged = QtCore.pyqtSignal(float, float)  # (x, y) ∈ [-1, 1]

    def __init__(self, size=170, parent=None):
        super().__init__(parent)
        self.setFixedSize(size, size)
        self._radius = size / 2 - 18
        self._knob_radius = 20
        self._x = 0.0
        self._y = 0.0
        self.setMouseTracking(True)

    def x(self):
        return self._x

    def y(self):
        return self._y

    def reset(self):
        self._reset()

    def _center(self):
        return QtCore.QPointF(self.width() / 2.0, self.height() / 2.0)

    def _set_from_pos(self, pos):
        c = self._center()
        dx = pos.x() - c.x()
        dy = pos.y() - c.y()
        dist = (dx * dx + dy * dy) ** 0.5
        if dist > self._radius:
            dx = dx / dist * self._radius
            dy = dy / dist * self._radius
        self._x = dx / self._radius
        self._y = -dy / self._radius  # 屏幕 y 向下，转为向上为正
        self.update()
        self.valueChanged.emit(self._x, self._y)

    def _reset(self):
        self._x = 0.0
        self._y = 0.0
        self.update()
        self.valueChanged.emit(0.0, 0.0)

    def paintEvent(self, event):
        p = QtGui.QPainter(self)
        p.setRenderHint(QtGui.QPainter.RenderHint.Antialiasing)
        c = self._center()
        # 方框
        p.setPen(QtGui.QPen(QtGui.QColor("#D2D2D7"), 1))
        p.setBrush(QtGui.QColor("#FFFFFF"))
        p.drawRoundedRect(0, 0, self.width() - 1, self.height() - 1, 12, 12)
        # 外圈
        p.setBrush(QtGui.QColor("#F5F5F7"))
        p.drawEllipse(c, self._radius, self._radius)
        # 十字参考线
        p.setPen(QtGui.QPen(QtGui.QColor("#E5E5EA"), 1))
        p.drawLine(QtCore.QPointF(c.x() - self._radius, c.y()), QtCore.QPointF(c.x() + self._radius, c.y()))
        p.drawLine(QtCore.QPointF(c.x(), c.y() - self._radius), QtCore.QPointF(c.x(), c.y() + self._radius))
        # 摇杆圆点
        knob = QtCore.QPointF(c.x() + self._x * self._radius, c.y() - self._y * self._radius)
        p.setPen(QtGui.QPen(QtGui.QColor("#D2D2D7"), 1))
        p.setBrush(QtGui.QColor("#0071E3"))
        p.drawEllipse(knob, self._knob_radius, self._knob_radius)

    def mousePressEvent(self, event):
        if event.button() == QtCore.Qt.MouseButton.LeftButton:
            self._set_from_pos(event.pos())

    def mouseMoveEvent(self, event):
        if event.buttons() & QtCore.Qt.MouseButton.LeftButton:
            self._set_from_pos(event.pos())

    def mouseReleaseEvent(self, event):
        self._reset()


class HJoystickWidget(QtWidgets.QWidget):
    """水平一维摇杆：左右拖动输出 -1..1（用于顺/逆时针旋转），松开回中。"""

    valueChanged = QtCore.pyqtSignal(float)

    def __init__(self, height=48, parent=None):
        super().__init__(parent)
        self.setFixedHeight(height)
        self.setMinimumWidth(160)
        self._knob_radius = 16
        self._value = 0.0
        self.setMouseTracking(True)

    def value(self):
        return self._value

    def reset(self):
        self._reset()

    def _track_rect(self):
        left = self._knob_radius + 4
        right = self.width() - self._knob_radius - 4
        return QtCore.QRectF(left, self.height() / 2.0 - 4.0, right - left, 8.0)

    def _set_from_x(self, x):
        r = self._track_rect()
        cx = (r.left() + r.right()) / 2.0
        half = r.width() / 2.0
        v = (x - cx) / half
        v = max(-1.0, min(1.0, v))
        self._value = v
        self.update()
        self.valueChanged.emit(v)

    def _reset(self):
        self._value = 0.0
        self.update()
        self.valueChanged.emit(0.0)

    def paintEvent(self, event):
        p = QtGui.QPainter(self)
        p.setRenderHint(QtGui.QPainter.RenderHint.Antialiasing)
        r = self._track_rect()
        cx = (r.left() + r.right()) / 2.0
        cy = self.height() / 2.0
        # 轨道
        p.setPen(QtCore.Qt.PenStyle.NoPen)
        p.setBrush(QtGui.QColor("#E5E5EA"))
        p.drawRoundedRect(r, 4, 4)
        # 中点标记
        p.setBrush(QtGui.QColor("#D2D2D7"))
        p.drawRect(QtCore.QRectF(cx - 1, cy - 10, 2, 20))
        # 滑块
        kx = cx + self._value * r.width() / 2.0
        p.setPen(QtGui.QPen(QtGui.QColor("#D2D2D7"), 1))
        p.setBrush(QtGui.QColor("#0071E3"))
        p.drawEllipse(QtCore.QPointF(kx, cy), self._knob_radius, self._knob_radius)

    def mousePressEvent(self, event):
        if event.button() == QtCore.Qt.MouseButton.LeftButton:
            self._set_from_x(event.pos().x())

    def mouseMoveEvent(self, event):
        if event.buttons() & QtCore.Qt.MouseButton.LeftButton:
            self._set_from_x(event.pos().x())

    def mouseReleaseEvent(self, event):
        self._reset()
