import os
import torch
import cv2
from ...processing import inference, io_utils
import cv2
import torch.nn.functional as F
from torchvision import transforms
from torchvision.io import read_image
from torchvision.transforms import ConvertImageDtype, Pad, Compose


def cos_sim(embedding1, embedding2):
    embedding1 = embedding1.unsqueeze(0) if embedding1.dim() == 1 else embedding1
    embedding2 = embedding2.unsqueeze(0) if embedding2.dim() == 1 else embedding2
    sim_tensor = F.cosine_similarity(embedding1, embedding2, dim=1)
    return sim_tensor.item()


def generate_detections(video_file, detector, stride=60, start=0):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    base_path = '../input_files/'
    cap = cv2.VideoCapture(base_path + video_file)

    cap.set(cv2.CAP_PROP_POS_FRAMES, start)
    f_num = start
    while True:
        ret, frame = cap.read()
        if not ret:
            print('Failure to read from file')
            break

        if (f_num - start) % stride == 0:
            detections = inference.detect_yolov4(frame, 0, detector, device)
            
            for i, box in enumerate(detections):
                x1, y1 = box[0], box[1]
                x2, y2 = box[0] + box[2], box[1] + box[3]
                cropped = frame[y1:y2, x1:x2]
                filename = f'../test_data/cropped_boxes/{f_num}_{i}.jpg'
                cv2.imwrite(filename, cropped)

        if stride <= 15:
            f_num += 1
        else:
            f_num += stride
            cap.set(cv2.CAP_PROP_POS_FRAMES, f_num)

    cap.release()


def crop_boxes(stride):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    yolov4 = inference.load_yolov4('YOLOv4.pth', device)

    primary = io_utils.get_queue_block()
    if len(primary) == 0:
        return 'No clips in the queue'

    for row in primary[:1]:
        video_file = row[1]
        print('detecting and embedding...')
        generate_detections(video_file, yolov4, stride=stride)


def test_similarity(base_path='../test_data/cropped_boxes'):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    extractor = inference.load_extractor('model.pth.tar-250', device)

    transform = transforms.Compose([transforms.Resize((256, 128)),
                                ConvertImageDtype(torch.float32)])
    with torch.no_grad():
        while True:
            file1 = input('Enter image 1: ')
            file2 = input('Enter image 2: ')

            imgs = [
                read_image(os.path.join(base_path, file1 + '.jpg')),
                read_image(os.path.join(base_path, file2 + '.jpg'))
            ]

            embeddings = []
            for img in imgs:
                if img.shape[0] == 4:
                    img = img[:3, :, :]
                img = transform(img)
                img = img.unsqueeze(0).to(device)
                embeddings.append(extractor(img))


            similarity = cos_sim(*embeddings)
            print(f'{file1} / {file2} similarity: {round(similarity, 3)}')


if __name__ == '__main__':
    command = input('Enter "c" to crop boxes or "s" to test similarity: ')
    if command.lower() == 'c':
        crop_boxes(stride=60)
    elif command.lower() == 's':
        test_similarity()
