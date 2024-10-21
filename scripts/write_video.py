import input_output as io_utils
import cv2


if __name__ == '__main__':
    colors = {'jimmy_smith': (252, 113, 3), 'jessica_owens': (252, 3, 211)}

    location = 'CP_Sacramento'
    timestamp = '2024-08-12_08_35_57'
    base_path = f'../intermediate_output/{location}_{timestamp}'
    trk_path = base_path + '_trk_data.hdf5'

    config = io_utils.get_config()
    cams = config['primary_cameras'] + config['secondary_cameras']

    _, all_trks = io_utils.get_trk_data(trk_path, cams, min_span=60)

    for c in config['primary_cameras']:
        trks = [trk for trk in all_trks.keys() if trk.startswith(c) and
                all_trks[trk].get('identity', False)]
        vid_path = f'../input_files/{location}_{timestamp}_{c.strip("c")}.mp4'
        cap = cv2.VideoCapture(vid_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        fw = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        fh = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        out = cv2.VideoWriter(f'../output_files/{location}_{timestamp}_'
                              + f'{c.strip("c")}_identified.mp4',
                              cv2.VideoWriter_fourcc(*'mp4v'), fps, (fw, fh))
        
        f_num = 0 
        while True:
            if f_num % 150 == 0:
                print(f_num)
            ret, frame = cap.read()
            if not ret:
                break

            for trk in trks:
                box = all_trks[trk]['detections'].get(f_num, False)
                if box:
                    box = [int(v) for v in box[:2]] + [int(sum(box[0:3:2])),
                                                       int(sum(box[1:4:2]))]
                    identity = all_trks[trk]['identity']
                    color = colors[identity]
                    cv2.rectangle(frame, box[:2], box[2:], color)
                    cv2.putText(frame, identity, (box[0], box[1]),
                                cv2.FONT_HERSHEY_PLAIN, 3, color, 2)
            out.write(frame)
            f_num += 1

        cap.release()
        out.release()