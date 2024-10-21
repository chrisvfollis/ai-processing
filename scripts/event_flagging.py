import csv

from utilities import (i_over_u, percent_in_polygon, read_detection_csv,
                       read_entryway_csv)


def flag_overlap_events(detection_data, iou_threshold=.05):
    '''
    Returns a list of frame indices. These are frames in which a detection
    overlaps with another detection above a certain IOU threshold.
    '''

    events = []

    for frame, detections in detection_data.items(): 
        for i in range(len(detections)-1):
            for j in range(len(detections)):
                if j > i:
                    i_o_u = i_over_u(detections[i], detections[j])
                    if i_o_u > iou_threshold:
                        events.append(frame)
                        break
    
    return events


def flag_entryway_events(detection_data, entryways, percent_threshold=.5):
    '''
    Flags entryway events: where a detected individual appears to be in one of
    the scene's entryways.
    '''
    events = {}
    for frame, detections in detection_data.items():
        for i in range(len(detections)):
            for entryway, points in entryways.items():
                percent_in_entrance = percent_in_polygon(detections[i], points)
                if percent_in_entrance > percent_threshold:
                    if frame in events:
                        events[frame].append([entryway] + detections[i][:4])
                    else:
                        events[frame] = [[entryway] + detections[i][:4]]
                    

    return events


def export_event_data(event_data, video):

    base_path = f'../intermediate_output/{video.split(".")[0]}'

    try:
        entryway_events = event_data['entryway_events']
        csv_path = base_path + '_entryway_events.csv'
        with open(csv_path, 'w', newline='') as file:
            csvwriter = csv.writer(file, delimiter=',')
            csvwriter.writerow(['frame', 'entryway', 'x', 'y', 'w', 'h'])
            for frame, events in entryway_events.items():
                for event in events:
                    csvwriter.writerow([frame] + event)
    except KeyError:
        print('No entryway event data provided')
    
    try:
        overlap_events = event_data['overlap_events']
        txt_path = base_path + '_overlap_events.txt'
        with open(txt_path, 'w') as file:
            file.write(','.join([str(f) for f
                                 in overlap_events]))
    except KeyError:
        print('No overlap event data provided')
    
    return 'Done'


if __name__ == '__main__':
    for cam in range(0, 6):
        workspace = 'CP_Sacramento'
        date_and_time = '2024-08-12_08_35_57'
        file_prefix = f'{workspace}_{date_and_time}_{cam}'
        config_prefix = f'{workspace}_cam{cam}'

        detection_path = f'../intermediate_output/{file_prefix}_detections.csv'
        entryway_path = f'../config/{config_prefix}_entryways.csv'
        entryways = read_entryway_csv(entryway_path)
        detection_data = read_detection_csv(detection_path)

        overlap_events = flag_overlap_events(detection_data)
        entryway_events = flag_entryway_events(detection_data, entryways)

        all_events = {'overlap_events': overlap_events,
                    'entryway_events': entryway_events}

        export_event_data(all_events, file_prefix)