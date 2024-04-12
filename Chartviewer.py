import sys
import numpy as np
from PyQt5.QtWidgets import QApplication, QWidget, QVBoxLayout
from PyQt5.QtChart import QChartView, QLineSeries, QScatterSeries,QChart
from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtGui import QPainter

class WaveformPlot(QWidget):
    def __init__(self):
        super().__init__()

        # self.setWindowTitle("Waveform Plot")
        # self.setGeometry(100, 100, 800, 600)

        layout = QVBoxLayout(self)

        self.chart1_view = QChartView(self)
        layout.addWidget(self.chart1_view)
        self.chart2_view = QChartView(self)
        layout.addWidget(self.chart2_view)

        self.chart1 = QChart()
        self.chart1.setTitle('X')
        self.chart1.setAnimationOptions(QChart.SeriesAnimations)
        self.chart1_view.setChart(self.chart1)

        self.series1 = QLineSeries()
        self.chart1.addSeries(self.series1)

        self.highlight_series1 = QLineSeries()
        self.highlight_series1.setColor(Qt.red)
        self.chart1.addSeries(self.highlight_series1)

        self.x_max = 50  # 初始显示的点数
        self.PredictLen = 5
        self.y_max = 600  # Y 轴范围

        self.chart1.createDefaultAxes()
        self.chart1.axisX().setRange(-self.x_max,self.PredictLen)
        self.chart1.axisY().setRange(0, self.y_max)


        self.Pos_data1 = np.zeros(self.x_max)  # 初始化信号数据，全部为零
        self.chart2 = QChart()
        self.chart2.setTitle('Y')
        self.chart2.setAnimationOptions(QChart.SeriesAnimations)

        self.chart2_view.setChart(self.chart2)

        self.series2 = QLineSeries()
        self.chart2.addSeries(self.series2)

        self.highlight_series2 = QLineSeries()
        self.highlight_series2.setColor(Qt.red)

        self.chart2.addSeries(self.highlight_series2)

        self.chart2.createDefaultAxes()
        self.chart2.axisX().setRange(-self.x_max,self.PredictLen)
        self.chart2.axisY().setRange(0, self.y_max)
      
        self.Pos_data2 = np.zeros(self.x_max)  # 初始化信号数据，全部为零


    def GUi_waveform(self,Pos_data1,Pos_data2):
        # 生成新的数据点，可以根据需要替换为你的数据生成逻辑


        self.series1.clear()
        for i in range(self.x_max):
            x, y = i-self.x_max, Pos_data1[i]
            self.series1.append(x, y)

        self.highlight_series1.clear()
        for i in range(self.PredictLen):
            x, y = i, Pos_data1[i+self.x_max]
            self.highlight_series1.append(x, y)
        # self.chart1_view.repaint()  # 或者 self.chart1_view.update()
        self.chart1_view.update()

        self.series2.clear()
        for i in range(self.x_max):
            x, y = i-self.x_max, Pos_data2[i]
            self.series2.append(x, y)

        self.highlight_series2.clear()
        for i in range(self.PredictLen):
            x, y = i, Pos_data2[i+self.x_max]
            self.highlight_series2.append(x, y)
        # self.chart2_view.repaint()  # 或者 self.chart2_view.update()
        self.chart2_view.update()


    def update_waveform(self,NewValx,NewValy):
        # 生成新的数据点，可以根据需要替换为你的数据生成逻辑
        self.Pos_data1 = np.roll(self.Pos_data1, -1)  # 向前滚动数据
        self.Pos_data1[-1] = NewValx

        self.series1.clear()
        for i in range(self.x_max):
            x, y = i-self.x_max, self.Pos_data1[i]
            self.series1.append(x, y)

        self.highlight_series1.clear()
        for i in range(self.PredictLen):
            x, y = i, self.Pos_data1[i]+120
            self.highlight_series1.append(x, y)
        # self.chart1_view.repaint()  # 或者 self.chart1_view.update()
        self.chart1_view.update()

        self.Pos_data2 = np.roll(self.Pos_data2, -1)  # 向前滚动数据
        self.Pos_data2[-1] = NewValy

        self.series2.clear()
        for i in range(self.x_max):
            x, y = i-self.x_max, self.Pos_data2[i]
            self.series2.append(x, y)

        self.highlight_series2.clear()
        for i in range(self.PredictLen):
            x, y = i, self.Pos_data2[i]+120
            self.highlight_series2.append(x, y)
        # self.chart2_view.repaint()  # 或者 self.chart2_view.update()
        self.chart2_view.update()
if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = WaveformPlot()
    window.show()
    sys.exit(app.exec_())
