import sys
import math
from PyQt5 import QtWidgets, QtCore, QtGui
from PyQt5.QtGui import QIcon, QPixmap

class MyWidget(QtWidgets.QWidget):
    def __init__(self, parent, size_factor = (3,3), pixmap = None):
        super().__init__()
        self.setWindowTitle('Auto Plot')
        self.setGeometry(30,30,600,400)
        self.label = QtWidgets.QLabel(self)
        if pixmap == None:
            self.pixmap = QPixmap('input/chosen_pic.png')
        else:
            self.pixmap = pixmap

        self.setFixedSize(self.pixmap.width(), self.pixmap.height())

        self.begin = QtCore.QPoint()
        self.end = QtCore.QPoint()

    def paintEvent(self, event):
        qp = QtGui.QPainter(self)
        qp.drawPixmap(self.rect(), self.pixmap)
        br = QtGui.QBrush(QtGui.QColor(100, 90, 10, 90))  
        qp.setBrush(br)   
        qp.drawRect(QtCore.QRect(self.begin, self.end))

        qp.drawLine(self.begin, self.end)
        qp.drawLine(self.end.x(),\
                    self.begin.y(),\
                    self.begin.x(),\
                    self.end.y())

    def mousePressEvent(self, event):
        self.begin = event.pos()
        self.end = event.pos()
        self.update()

    def mouseMoveEvent(self, event):
        self.end = event.pos()
        self.update()

    def mouseReleaseEvent(self, event):
        # self.begin = event.pos()
        self.end = event.pos()
        # self.update()