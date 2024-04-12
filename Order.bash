# source ~/.bashrc


using python 3.6
conda activate CondaDrone36


uic:

pyuic5 ./Serial/heart.ui -o ./Serial/heartuic.py

#essential
in heartuic.py


alter graphicsView to StackedWidget


#use os
os.system('pip install -U ultralytics')



# for yolo 3.9
conda activate yolo5py39
cd Drone2
python detect.py --source 0 --weights ../D-Drone_v2-main/YOLOv5/weights/best.pt 

python ./yolov5/detectdrone.py

cd Drone2/yolov4/darknet
python ./yolov4/darknet/cvdronedetect.py 

./darknet detector test <path_to_data_file> <path_to_yolov4_cfg> <path_to_weights> <path_to_image>

python detect.py --source 0 --weights ../D-Drone_v2-main/YOLOv4/weights/drone_1000.weights 



# for yolo 3.8

conda activate yolo38
python ./Serial/pyserial.py 