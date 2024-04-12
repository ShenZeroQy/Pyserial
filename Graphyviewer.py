import sys
import numpy as np
from PyQt5.QtWidgets import QApplication, QVBoxLayout, QWidget
import pyqtgraph as pg
from pyqtgraph import PlotWidget

class WaveformPlot(QWidget):
    def __init__(self):
        super().__init__()

        self.setWindowTitle("PyQtGraph Dual-Channel WaveformPlot Example")
        # self.setGeometry(100, 100, 800, 600)

        self.layout = QVBoxLayout()
        self.setLayout(self.layout)

        # 创建两个绘图小部件并添加到布局中
        self.plot_widget_figX = PlotWidget(self)
        self.plot_widget_figY = PlotWidget(self)
        self.layout.addWidget(self.plot_widget_figX)
        self.layout.addWidget(self.plot_widget_figY)

        # 初始化数据和指针
        self.TrainLen=50
        TestLen=5
        self.plot_data_channelX1 = np.zeros(self.TrainLen)
        self.plot_data_channelY1 = np.zeros(self.TrainLen)
        self.ptr = 0

        # 创建曲线并连接到数据
        self.curve_channelX1 = self.plot_widget_figX.plot(pen='g')
        # self.curve_channelX2 = self.plot_widget_figX.plot(pen='b')

        self.curve_channelY1 = self.plot_widget_figY.plot(pen='b')

        # 设置定时器以实时更新波形
        # self.timer = pg.QtCore.QTimer(self)
        # self.timer.timeout.connect(self.update_plot)
        # self.timer.start(5)  # 每50毫秒更新一次

    def update_plot(self,X,Y):
 

        self.curve_channelX1.setData(X)
        # self.curve_channelX2.setData(Y)

        self.curve_channelY1.setData(Y)

if __name__ == '__main__':
    app = QApplication(sys.argv)
    WaveformPlot = WaveformPlot()
    WaveformPlot.show()
    sys.exit(app.exec_())
