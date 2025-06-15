import random
import cv2
from utilities import general_utils as utils
from datetime import datetime


def random_bbox(
        img_width: int = 3840,
        img_height: int = 2160,
        min_area: int = 40000,
        max_area: int = 2250000,
        aspect_ratio_range=(0.5, 2.0)
    ) -> tuple[int, int, int, int]:
    '''
    Generate a random bounding box (x, y, w, h) that fits within the image dimensions
    and has an area between min_area and max_area. The aspect ratio (w/h) is sampled
    uniformly within aspect_ratio_range to get more varied shapes.
    
    Returns:
        tuple[int, int, int, int]: (x, y, w, h), the bounding rectangle.
    '''
    assert img_width * img_height >= min_area, 'Image is too small for the minimum area.'
    
    area = random.randint(min_area, min(max_area, img_width * img_height))
    
    min_ar, max_ar = aspect_ratio_range
    ar = random.uniform(min_ar, max_ar)
    
    w = int(round((area * ar) ** 0.5))
    h = int(round((area / ar) ** 0.5))
    
    w = min(w, img_width)
    h = min(h, img_height)
    
    actual_area = w * h
    
    if actual_area < min_area:
        if w < h and w < img_width:
            w = min(img_width, int(min_area / h))
        elif h < img_height:
            h = min(img_height, int(min_area / w))
        w = min(w, img_width)
        h = min(h, img_height)
    
    x = random.randint(0, img_width - w)
    y = random.randint(0, img_height - h)
    
    return x, y, w, h


def save_imgs(video_path: str = 'files/input/2025-04-18_10-25-01_0.mp4'):
    start = datetime.now()
    cap = cv2.VideoCapture(video_path)
    resolution, _, fps, f_tot = utils.get_video_info(cap, release=False)

    step = int(f_tot/75)

    f_nums = range(0, f_tot, step)
    for f_num in f_nums:
        cap.set(cv2.CAP_PROP_POS_FRAMES, f_num)
        ret, frame = cap.read()
        if not ret:
            continue
        
        x, y, w, h = random_bbox(*resolution)
        img = frame[y:y+h, x:x+w]
        cv2.imwrite(f'files/output/event_imgs/{f_num}.jpg', img)

    cap.release()
    end = datetime.now()
    total = (end - start).total_seconds()/60
    print(f'minutes: {total}')


if __name__ == '__main__':
    save_imgs()
