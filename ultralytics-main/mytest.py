from ultralytics import YOLO

if __name__ == '__main__':
    model = YOLO('E:\\zqg\\yolov8project\\ultralytics-main\\runs\\detect\\train768\\weights\\best.pt')  #自己训练结束后的模型权重
    model.val(data='E:\\zqg\\yolov8project\\ultralytics-main\\datasets\\mydate\\mydate.yaml',
              split='val',
              imgsz=640,
              batch=16,
              save_json=True, # if you need to cal coco metrice
              project='runs/val',
              name='exp',
              )
