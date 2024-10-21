import torch
import input_output as io_utils
from object_detection import load_yolov4, detect_yolov4, process_clip
import cv2

def detection_skim(file, model, stride=60):
    f_num = 0
    footage_path = '../input_files/'
    cap = cv2.VideoCapture(footage_path + file)
    cap.set(cv2.CAP_PROP_POS_FRAMES, f_num)

    while True:
        ret, frame = cap.read()
        if not ret:
            cap.release()
            return False

        if f_num % stride == 0:
            det_xywhc = detect_yolov4(frame, 0, model, device,
                                      conf_thresh=.75)
            if len(det_xywhc) > 0:
                cap.release()
                return True

        f_num += stride
        cap.set(cv2.CAP_PROP_POS_FRAMES, f_num)


if __name__ == '__main__':
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    yolov4 = load_yolov4('YOLOv4.pth', device)

    while True:
        primary = io_utils.get_queue_block('../appdata/data.db',
                                        designation='primary')
        
        detections = False
        for row in primary:
            file = row[1]
            if detection_skim(file, yolov4):
                detections = True
                break
        
        if detections == True:
            for row in primary:
                file = row[1]
                frame_data = process_clip(file, yolov4, stride=3)
                io_utils.write_detection_csv(frame_data, file)

                
