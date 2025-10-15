from ultralytics import YOLO

if __name__ == '__main__':
    # build from YAML and transfer weights
    model = YOLO('F:\\MCAM-Net\\ultralytics-main\\MCAM-Net.yaml').load('F:\\MCAM-Net\\ultralytics-main\\yolov8n.pt')
    # Train the model
    model.train(data='F:\\MCAM-Net\\ultralytics-main\\datasets\\mydate\\mydate.yaml', workers=0, epochs=100, imgsz=640)
