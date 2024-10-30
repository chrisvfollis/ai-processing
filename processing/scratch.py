import io_utils
import utilities



def check_overlap(track, cam):
    trk_path = '../intermediate_output/2024-08-12_08-35-57_trk_data.hdf5'
    _, all_trks = io_utils.get_trk_data(trk_path, [cam])

    config = io_utils.get_config()
    entryways = config['entryways'][cam].values()

    f1 = min(all_trks[track]['detections'].keys())

    detection = all_trks[track]['detections'][f1]
    print(detection)

    for entryway in entryways:
        print(utilities.percent_in_polygon(detection, entryway))
    
    print(all_trks[track]['trk_span'])

if __name__ == '__main__':
    check_overlap('c0_trk0', 'c0')
