# To fix the auroty permision denyed
#https://www.lmlphp.com/user/365769/article/item/8231724/
#file written to /etc/udev/rules.d/TTYRs422.rules

import os
import sys
import serial
import struct  # use struct
import serial.tools.list_ports
from PyQt5 import QtWidgets
from PyQt5.QtWidgets import QMessageBox
from PyQt5.QtCore import QTimer
from PyQt5 import QtGui
from heartuic import Ui_Form
# from heart import Ui_Form
# import heartuic
from PyQt5.QtWidgets import QApplication, QMainWindow, QLabel, QPushButton, QVBoxLayout, QWidget , QStackedWidget
from PyQt5.QtGui import QPixmap, QImage, QColor

import numpy as np
import cv2
import binascii
#old version


current_dir = os.getcwd()
print("Current working directory:", current_dir)

#Extend dir
sys.path.append('./')





# from CpyProt import ProtPy as P 



from Serial import Graphyviewer as QtchartPloter
from Predictor import ATPESN as ESN


currentdir= os.getcwd()
class Pyqt5_Serial(QtWidgets.QWidget, Ui_Form):
    # 初试化程序
    def __init__(self):
        super(Pyqt5_Serial, self).__init__()
        self.setupUi(self)
        self.Serial_init()
        self.CV_init()
        self.setWindowTitle("PoweredByQyShen")
        self.setWindowIcon(QtGui.QIcon('Serial/icon/Qy.jpeg'))
        print('ico dir:'+currentdir+'Serial/icon/Qy.jpeg')
        self.ser = serial.Serial()
        self.port_check()
        # 接收数据和发送数据数目置零
        self.data_num_received = 0
        self.lineEdit.setText(str(self.data_num_received))
        self.data_num_sended = 0
        self.lineEdit_2.setText(str(self.data_num_sended))
        # self.Port_send()
        #初始化轨迹GUI
        self.Init_polter()
        self.i=0


    # 引用控件
    def Serial_init(self):
        #load Cpython Protocl module
        # self.Protocl=P.CPySerialProtocl()
        # #测试Cpython 通信协议
        # P.Test_Prot()

        # 串口检测按钮
        self.s1__box_1.clicked.connect(self.port_check)

        # 串口信息显示
        self.s1__box_2.currentTextChanged.connect(self.port_imf)

        # 打开串口按钮
        self.open_button.clicked.connect(self.port_open)

        # 关闭串口按钮
        self.close_button.clicked.connect(self.port_close)

        # 发送数据按钮
        self.s3__send_button.clicked.connect(self.data_send_windows)

        # 定时发送数据
        self.Serial_timer_sender = QTimer()
        self.Serial_timer_sender.timeout.connect(self.data_send_windows)
        self.Serial_timer_sender_cb.stateChanged.connect(self.data_send_timer)

        # 定时器接收数据
        self.Serial_timer_receiver = QTimer(self)
        # self.Serial_timer_receiver.timeout.connect(self.data_receive)

        #receive and  decode
        self.Serial_timer_receiver.timeout.connect(self.data_receive_and_decode)
        

        # 清除发送窗口
        self.s3__clear_button.clicked.connect(self.send_data_clear)

        # 清除接收窗口
        self.s2__clear_button.clicked.connect(self.receive_data_clear)

    # 串口检测
    def port_check(self):
        # 检测所有存在的串口，将信息存储在字典中
        self.Com_Dict = {}
        port_list = list(serial.tools.list_ports.comports())
        self.s1__box_2.clear()
        for port in port_list:
            self.Com_Dict["%s" % port[0]] = "%s" % port[1]
            self.s1__box_2.addItem(port[0])
        if len(self.Com_Dict) == 0:
            self.state_label.setText(" 无串口")

    # 串口信息
    def port_imf(self):
        # 显示选定的串口的详细信息
        imf_s = self.s1__box_2.currentText()
        if imf_s != "":
            self.state_label.setText(self.Com_Dict[self.s1__box_2.currentText()])

    # 打开串口
    def port_open(self):
        self.ser.port = self.s1__box_2.currentText()
        self.ser.baudrate = int(self.s1__box_3.currentText())
        self.ser.bytesize = int(self.s1__box_4.currentText())
        self.ser.stopbits = int(self.s1__box_6.currentText())
        self.ser.parity = self.s1__box_5.currentText()

        #检测异常
        try:
            self.ser.open()
        except Exception as e:
            # 捕捉到其他未知异常
            print("发生异常：", str(e))
            QMessageBox.critical(self, "Port Error", "此串口不能被打开！"+str(e))
            # self.ser=serial.Serial(self.s1__box_2.currentText(), int(9600), timeout=0.5)
            return None

        # 打开串口接收定时器，周期为2ms
        self.Serial_timer_receiver.start(2)

        if self.ser.isOpen():
            self.open_button.setEnabled(False)
            self.close_button.setEnabled(True)
            self.formGroupBox1.setTitle("串口状态（已开启）")
            self.plainTextEdit.setPlainText("ready")


    # 关闭串口
    def port_close(self):
        self.Serial_timer_receiver.stop()
        self.Serial_timer_sender.stop()
        try:
            self.ser.close()
        except:
            pass
        self.open_button.setEnabled(True)
        self.close_button.setEnabled(False)
        self.lineEdit_3.setEnabled(True)
        # 接收数据和发送数据数目置零
        self.data_num_received = 0
        self.lineEdit.setText(str(self.data_num_received))
        self.data_num_sended = 0
        self.lineEdit_2.setText(str(self.data_num_sended))
        self.formGroupBox1.setTitle("串口状态（已关闭）")

    # 发送数据
    def data_send_windows(self):
        if self.ser.isOpen():
            input_s = self.s3__send_text.toPlainText()
            if input_s != "":
                # 非空字符串
                if self.hex_send.isChecked():
                    # hex发送
                    input_s = input_s.strip()
                    send_list = []
                    while input_s != '':
                        try:
                            num = int(input_s[0:2], 16)
                        except ValueError:
                            QMessageBox.critical(self, 'wrong data', '请输入十六进制数据，以空格分开!')
                            return None
                        input_s = input_s[2:].strip()
                        send_list.append(num)
                    input_s = bytes(send_list)
                else:
                    # ascii发送
                    input_s = (input_s + '\r\n').encode('utf-8')

                num = self.ser.write(input_s)
                self.data_num_sended += num
                self.lineEdit_2.setText(str(self.data_num_sended))
        else:
            pass

    # 接收数据
    def data_receive(self):
        try:
            num = self.ser.inWaiting()
        except:
            self.port_close()
            return None
        if num > 0:
            data = self.ser.read(num)
            num = len(data)
            # hex显示
            if self.hex_receive.checkState():
                out_s = ''
                for i in range(0, len(data)):
                    out_s = out_s + '{:02X}'.format(data[i]) + ' '
                self.s2__receive_text.insertPlainText(out_s)
            else:
                # 串口接收到的字符串为b'123',要转化成unicode字符串才能输出到窗口中去
                self.s2__receive_text.insertPlainText(data.decode('iso-8859-1'))

            # 统计接收字符的数量
            self.data_num_received += num
            self.lineEdit.setText(str(self.data_num_received))

            # 获取到text光标
            textCursor = self.s2__receive_text.textCursor()
            # 滚动到底部
            textCursor.movePosition(textCursor.End)
            # 设置光标到text中去 
            self.s2__receive_text.setTextCursor(textCursor)
        else:
            pass
    # 接收数据并解码
    def data_receive_and_decode(self):
        try:
            num = self.ser.inWaiting()
        except:
            self.port_close()
            return None
        if num > 0:
            # Read data from the serial port
            serial_data = self.ser.read(20+8)  # Adjust the buffer size as necessary 20=16+4=4*4+4
            
            #GUI show
            out_s = ''
            for i in range(0, len(serial_data)):
                out_s = out_s + '{:02X}'.format(serial_data[i]) + ' '
            self.s2__receive_text.setPlainText('frame:'+out_s+'\n')

            if (serial_data and 1):
                decoded_data = self.decode_serial_data(serial_data.hex())
                #show decode result
                # self.plainTextEdit.setPlainText(str(decoded_data))
                while (decoded_data == None):#Lock and read
                    print('trigger of error frame')
                    serial_data=serial_data+self.ser.read(1) #read one more                    
                    #show decode result
                    out_s = ''
                    for i in range(0, len(serial_data)):
                        out_s = out_s + '{:02X}'.format(serial_data[i]) + ' '
                    self.s2__receive_text.insertPlainText('frame:'+out_s+'\n')

                    print('fixing'+out_s)
                    decoded_data = self.decode_serial_data(serial_data.hex())
                #docode success 
                # print(decoded_data)
                self.plainTextEdit.setPlainText(str(decoded_data))
                self.GUI_Trace_and_predict(decoded_data)

            # 统计接收帧的数量
            self.data_num_received += 1
            if(self.data_num_received>64000):
                self.data_num_received = 0
            self.lineEdit.setText(str(self.data_num_received))
            
        
        else:
            #mimic 模拟串口启动
            # self.mimic_serial_ploter(self.i)
            self.i=self.i+1
    # 定时发送数据
    def data_send_timer(self):
        if self.Serial_timer_sender_cb.isChecked():
            self.Serial_timer_sender.start(int(self.lineEdit_3.text()))
            self.lineEdit_3.setEnabled(False)
        else:
            self.Serial_timer_sender.stop()
            self.lineEdit_3.setEnabled(True)
   
    # 清除显示
    def send_data_clear(self):
        self.s3__send_text.setText("")

    def receive_data_clear(self):
        self.s2__receive_text.setText("")

    def Port_send(self,ts=12, ink=-2,dob=0.83):
        # F,F_len=self.Protocl.Serial_FrameData( self.Protocl.Get_timestamp,ink,dob)
        F,F_len=self.Protocl.Serial_FrameData(ts,ink,dob)

        # self.plainTextEdit.setPlainText('Len:'+str(F_len)+' F:'+str(self.Protocl.hexFrame()))
        if (self.checkBox_Taj_1.isChecked()):
            self.textBrowser_Taj.setPlainText(str(F_len)+'bytes:'+str(self.Protocl.hexFrame()))
            if(self.checkBox_Taj_2.isChecked()):
                self.Taj_Dirct_send(F[0:F_len])
    def Taj_Dirct_send(self,input_s):
        if self.ser.isOpen():
            num = self.ser.write(input_s)
            self.label_Taj.setText(str(num)+" has been written")
        else:
            self.label_Taj.setText("Please Open Serial Port")


    def decode_serial_data(self,data):
        # Convert hex string to bytes
        bytes_data = bytes.fromhex(data)

        # Find frame boundaries (start and end)
        start_index = bytes_data.rfind(b'\xAA\x55')
        end_index = bytes_data.rfind(b'\x0B\x0D')
        print(start_index)
        print(end_index)
        if start_index != -1 and end_index != -1 and end_index>start_index and end_index-start_index ==10:
            # frame = bytes_data[2:-2]#remove head and tail
            frame= bytes_data[start_index+2:end_index] 
            # decoded_frame = {
            #     # 'data': frame[0:16],  # Extracting the payload bytes
            #     # You might need to decode the payload bytes according to your protocol
            #     # For example, if it's IEEE 754 floating point, you can use struct.unpack
            #     # 'f' stands for float, 'd' stands for double
            #     # 'I' stands for unsigned integer, you may need to adjust according to your protocol
            #     'value1A': struct.unpack('<f', frame[0:4])[0], # Example decoding float
            #     'value1X': struct.unpack('<f', frame[4:8])[0], # Example decoding float

            #     'value1TS': struct.unpack('<i', frame[8:12])[0],
            #     'value2A': struct.unpack('<f', frame[12:16])[0],
            #     'value2X': struct.unpack('<f', frame[16:20])[0],
            #     'value2TS': struct.unpack('<i', frame[20:24])[0]

            #     # 'value4': struct.unpack('<f', frame[14:18])[0]     
            # }
            decoded_frame = (
                # 'f' stands for float, 'd' stands for double
                # 'I' stands for unsigned integer, you may need to adjust according to your protocol
                struct.unpack('<f', frame[0:4])[0],  # value1A
                struct.unpack('<f', frame[4:8])[0],  # value1X
                struct.unpack('<i', frame[8:12])[0], # value1TS
                struct.unpack('<f', frame[12:16])[0], # value2A
                struct.unpack('<f', frame[16:20])[0], # value2X
                struct.unpack('<i', frame[20:24])[0]  # value2TS
            )
            # decoded_frames.append(decoded_frame)
            return decoded_frame
        else:
            return None


    def encode_serial_data(self,ts,PosDelay):
        #head 
        data = b'\xAA\x55'+struct.pack('<i',ts)
        #data
        for Pos in PosDelay:
            Tof=struct.pack('<i',Pos[0])
            # Fx=struct.pack('<f',Pos[1])
            # Fy=struct.pack('<f',Pos[2])
            Fx=struct.pack('<f',1.0)
            Fy=struct.pack('<f',1.0)
            data = data + Tof + Fx +Fy

        # tail
        data =data + b'\x0B\x0D'
        hex_string = ' '.join(format(byte, '02X') for byte in data)

        # print("Sending Frame:", hex_string)

        # binary_data_length = len(data)
        # print("Sending Frame length:", binary_data_length)
        return data,hex_string

    def GUI_Trace_and_predict(self,decoded_data):
        # print(decoded_data)
        tsX=int(decoded_data[2])
        tsY=int(decoded_data[5])
        ts=tsX
        self.ESN.Sig.Push_back(decoded_data[0],decoded_data[1],int(decoded_data[2]),decoded_data[3],decoded_data[4],int(decoded_data[5]))
        ax,ay=self.ESN.Do_Prdict()
        self.WaveViewer.update_plot(ax.reshape(-1),ay.reshape(-1))# dim 1
        self.label_Taj.setText(str(self.ESN.TrainMSE))
        
        self.lineEdit_Taj_TS.setText(str(ts))
        self.lineEdit_Taj_x.setText(str(ax[60]))
        self.lineEdit_Taj_y.setText(str(ay[60]))
        DelayPos=[(i,ax[60+i],ay[60+i]) for i in range(2)]
        if (self.checkBox_Taj_1.isChecked()):
            Frame,FrameHex=self.encode_serial_data(ts,DelayPos)
            self.textBrowser_Taj.setPlainText(FrameHex)
            if(self.checkBox_Taj_2.isChecked()):
                self.ser.write(Frame)


##############################################################33cv
    def CV_init(self):
        #TV timer
        self.TV_timer = QTimer(self) 
        self.TV_timer.stop()
        self.TV_timer.timeout.connect(self.timerEvent)
        self.DT=False

        self.CV_cap=None
        self.CV=False
        # 串口检测按钮
        self.pushButton_TV_camera.clicked.connect(self.start_video)
        self.pushButton_TV_LoadModule.clicked.connect(self.LoadModule)

        # self.checkBox_TV_DroneDetection.isChecked()
        # self.checkBox_TV_AutoRecord.isChecked()
    def LoadModule(self):
        # self.DT=True
        # Detector.Init_Detector()
        # self.Init_polter()
        pass

    def start_video(self):
        #open
        if self.CV==False:
            self.pushButton_TV_camera.setText('close')
            self.CV=True

            if self.CV_cap is None:
                self.CV_cap=cv2.VideoCapture(0)
            if not self.CV_cap.isOpened():
                print("No camera opened ")
                exit()
            if not self.CV_cap.isOpened():
                return

            self.TV_timer.start(500)# Update frame every 33 milliseconds (30 fps)
            # self.TV_timer.timeout.connect(self.timerEvent)
        #close
        else:
            self.pushButton_TV_camera.setText('open')
            self.CV=False
            self.label_TV_face.setText('请打开摄像头')
            self.TV_timer.stop()
            # self.TV_timer.timeout.connect(self.timerEvent)
        self.label_TV.setText('TVState:'+str(self.CV))

    def timerEvent(self):
        if(self.CV):
            ret, frame= self.CV_cap.read()  
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB
            h, w, ch = frame.shape
            bytes_per_line = ch * w
            q_img = QImage(frame.data, w, h, bytes_per_line, QImage.Format_RGB888)
            pixmap = QPixmap.fromImage(q_img)
            self.label_TV_face.setPixmap(pixmap)

    def closeEvent(self, event):
        if self.CV_cap is not None:
            self.CV_cap.release()
        super().closeEvent(event)
#######################################################################polt Viewer
    def mimic_serial_ploter(self,i):#simulation 
        oumig=0.1
        mimicFrame=(np.sin(oumig*i),-np.sin(oumig*i),i%60,10.0,-10.0,i%60)
        self.GUI_Trace_and_predict(mimicFrame)
    def Init_polter(self):
        self.WaveViewer=QtchartPloter.WaveformPlot()
        self.StackedWidget_Taj_1.addWidget(self.WaveViewer)
        self.ESN=ESN.EchoStateNetwork()
        # self.StackedWidget_Taj_1.setCurrentWidget(WaveViewer)

        # self.CP2=QtchartPloter.WaveformPlot('Y')
        # self.StackedWidget_Taj_2.addWidget(self.CP2)
        # self.StackedWidget_Taj_2.setCurrentWidget(CP2)
    def Set_ploter(self,TS,Px,Py):
        self.ESN.Sig.Push_back(Px,Py,TS%65535)
        ax,ay=self.ESN.Do_Prdict()
        self.WaveViewer.update_plot(ax.reshape(-1),ay.reshape(-1))# dim 1
        # self.WaveViewer.update_waveform(Px,Py)


        self.lineEdit_Taj_TS.setText(str(TS))
        self.lineEdit_Taj_x.setText(str(Px))
        self.lineEdit_Taj_y.setText(str(Py))
        self.Port_send(TS,int(Px),Py)


    # 程序入口
if __name__ == '__main__':
    app = QtWidgets.QApplication(sys.argv)
    myshow = Pyqt5_Serial()
    myshow.show()
    sys.exit(app.exec_())
