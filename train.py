from ultralytics import YOLO #YOLOをインポート

model = YOLO('yolov5s.pt') #モデルの読み込み

model.train(data='dataset.yaml', epochs=100, imgsz=640) #学習の実行