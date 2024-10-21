import numpy as np
import bisect
import input_output as io_utils
from utilities import percent_in_polygon, is_coincident
import subprocess
import sys
import tensorflow as tf
from facial_recognition import facial_recognition
import datetime


def global_headcounts(all_frames, all_trks, prev_headcount=0):
    headcount = prev_headcount
    headcounts = dict(zip(all_frames, [0 for _ in all_frames]))

    entry_frames = sorted([trk['entry'] for trk in all_trks.values() if trk['entry']])
    exit_frames = sorted([trk['exit'] for trk in all_trks.values() if trk['exit']])

    prev_entry, prev_exit = 0, 0
    for frame in all_frames:
        entry_bound = bisect.bisect_right(entry_frames, frame)
        exit_bound = bisect.bisect_right(exit_frames, frame)

        headcount += len(entry_frames[prev_entry:entry_bound])
        headcount -= len(exit_frames[prev_exit:exit_bound])

        headcounts[frame] = headcount

        prev_entry, prev_exit = entry_bound, exit_bound
    
    return headcounts


def headcount_sections(all_trks, headcounts):
    section, sections = 0, {}
    frames = sorted(headcounts.keys())
    
    sections[section] = {'span': [frames[0]], 'tracks': []}

    prev_count = headcounts[frames[0]]
    for frame in frames:
        count = headcounts[frame]
        if count != prev_count:
            prev_count = count
            sections[section]['span'].append(frame - 1)
            section += 1
            sections[section] = {'span': [frame], 'tracks': []}

    sections[section]['span'].append(frames[-1])

    for section in sections.keys():
        for trk, data in all_trks.items():
            if is_coincident(sections[section]['span'], data['trk_span']):
                sections[section]['tracks'].append(trk)
    return sections


def flag_entryway_events(all_trks, entryways, threshold=.45):

    for id, trk in all_trks.items():
        cam = id.split('_')[0]

        start, end = trk['trk_span']
        detections = [trk['detections'][start][:4],
                    trk['detections'][end][:4]]
        keys = ['entry', 'exit']
        
        for i in range(2):
            for points in entryways[cam].values():
                pcnt_in_entryway = percent_in_polygon(detections[i], points)
                if pcnt_in_entryway > threshold:
                    trk[keys[i]] = trk['trk_span'][i]
                    break
                else:
                    trk[keys[i]] = None

    return all_trks


def filter_entryway_stays(all_trks, entryways, threshold=.5):

    remove = []
    for id, trk in all_trks.items():
        if (not trk['entry']) or (not trk['exit']):
            continue
        cam = id.split('_')[0]

        detections = trk['detections']
        frames = sorted(detections.keys())
        
        start = frames[0]
        for points in entryways[cam].values():
            pcnt_in_entryway = percent_in_polygon(detections[start], points)
            if pcnt_in_entryway > threshold:
                break

        for f in frames:
            if (f - start) % 20 == 0:
                pcnt_in_entryway = percent_in_polygon(detections[f], points)
                if pcnt_in_entryway < threshold:
                    break
            elif f == frames[-1]:
                remove.append(id) 
    for id in remove:
        del all_trks[id]
    return all_trks


def identify_tracks(all_trks, subset, min_span=0):
    trks = [trk for trk in subset if (all_trks[trk]['trk_span'][1]
                                      - all_trks[trk]['trk_span'][0])
                                      >= min_span]
    for trk in trks:
        cam = trk.split('_')[0].strip('c')
        vid_file = f'{location}_{timestamp}_{cam}.mp4'
        
        identity, _ = facial_recognition(vid_file, all_trks[trk])
        if identity:
            identity = '_'.join(identity.split('.')[0].split('_')[:2])
            print(f'{trk} direct ID: {identity}')
        all_trks[trk]['identity'] = identity

    return all_trks


def logical_identification(all_trks, section, headcount):
    '''
    Attempts to identify any unidentified tracks by using logic to
    determine if there are any already identified tracks that they
    must be associated with.
    '''

    def _same_cam_coincident(all_trks, trk1, trk2):
        '''
        Checks whether the tracks coincide with one another if they are from
        the same camera. If this is the case, then the tracks cannot have the
        same identity. 
        '''

        if trk1.split('_')[0] != trk2.split('_')[0]:
            return False
        return is_coincident(all_trks[trk1]['trk_span'],
                             all_trks[trk2]['trk_span'])

    def _entry_after_start(all_trks, trk1, trk2):
        '''
        Checks if either track enters after the other has already started. If
        this is the case, then the tracks cannot have the same identity. 
        '''

        t1_data = all_trks[trk1]
        t2_data = all_trks[trk2]

        t1_entry = t1_data.get('entry', None)
        t2_entry = t2_data.get('entry', None)

        if t1_entry and (t1_entry > t2_data['trk_span'][0]):
            return True
        elif t2_entry and (t2_entry > t1_data['trk_span'][0]):
            return True
        else:
            return False
    
    def _exit_before_end(all_trks, trk1, trk2):
        '''
        Checks if either track exits before the other ends. If this is the
        case, then the tracks cannot have the same identity. 
        '''

        t1_data = all_trks[trk1]
        t2_data = all_trks[trk2]

        t1_exit = t1_data.get('exit', None)
        t2_exit = t2_data.get('exit', None)

        if t1_exit and (t1_exit < t2_data['trk_span'][1]):
            return True
        elif t2_exit and (t2_exit < t1_data['trk_span'][1]):
            return True
        else:
            return False

    total = len(section['tracks'])
    identified = 0
    identities = {}
    # print(section['tracks'])
    possibilities = {}
    for trk1 in section['tracks']:
        if not all_trks[trk1].get('identity', False):
            for trk2 in section['tracks']:
                if trk2 == trk1:
                    continue

                args_ = [all_trks, trk1, trk2]
                if not (_same_cam_coincident(*args_)
                        or _entry_after_start(*args_)
                        or _exit_before_end(*args_)):
                    possibilities.setdefault(trk1, []).append(trk2)
        else:
            identity = all_trks[trk1]['identity']
            identities.setdefault(identity, []).append(trk1)
            identified += 1
            if identified == total:
                return all_trks
    
    one_possibility = [k for k, v in possibilities.items() if len(v) == 1]
    while len(one_possibility) > 0:
        for trk in one_possibility:
            link = possibilities[trk][0]
            if all_trks[link].get('identity', None):
                all_trks[trk]['identity'] = all_trks[link]['identity']
                print(f'{trk} ID by one possibility: {all_trks[link]["identity"]}')
            del possibilities[trk]

            for trk2, data in possibilities.items():
                if (trk2 != trk) and (link in data) and (len(data) != 1):
                    idx = data.index(link)
                    del data[idx]
        one_possibility = [k for k, v in possibilities.items() if len(v) == 1]
    # print(possibilities)

    candidates = {}
    for id, trks in identities.items():
        for k, v in possibilities.items():
            if set(v) == set(v + trks):
                candidates.setdefault(id, []).append(k)

    newly_identified = []
    for id, trks in candidates.items():
        for trk in trks:
            i = trks.index(trk)
            filtered = [x for i2, x in enumerate(trks) if i2 != i]
            if set(possibilities[trk]) == set(possibilities[trk] + filtered):
                newly_identified.append(trk)
                all_trks[trk]['identity'] = id
                print(f'{trk} ID by elimination: {id}')
    
    # returns c1_trk5 instead of c0_trk10:
    for id, trks in candidates.items():
        candidates[id] = [trk for trk in trks if trk not in newly_identified]
    for id, trks in candidates.items():
        newly_identified = []
        for trk in trks:
            i = trks.index(trk)
            filtered = [x for i2, x in enumerate(trks) if i2 != i]
            if set(possibilities[trk]) == set(possibilities[trk] + filtered):
                newly_identified.append(trk)
                all_trks[trk]['identity'] = id
                print(f'{trk} ID by elimination: {id}')

    return all_trks


if __name__ == '__main__':

    physical_devices = tf.config.list_physical_devices('GPU')
    if physical_devices:
        try:
            tf.config.experimental.set_visible_devices(physical_devices[0], 'GPU')
            print("Using GPU: ", physical_devices[0], flush=True)
        except RuntimeError as e:
            print(e, flush=True)
    else:
        print("No GPU available", flush=True)

    start = datetime.datetime.now()

    stride = 3

    location = 'CP_Sacramento'
    timestamp = '2024-08-12_08_35_57'
    base_path = f'../intermediate_output/s{stride}_{location}_{timestamp}'
    trk_path = base_path + '_trk_data.hdf5'

    config = io_utils.get_config()
    primary_cams = config['primary_cameras']
    secondary_cams = config['secondary_cameras']
    entryways = config['entryways']

    metadata, all_trks = io_utils.get_trk_data(trk_path, primary_cams, min_span=60)
    _, secondary_trks = io_utils.get_trk_data(trk_path, secondary_cams, min_span=60)
    io_utils.update_identities(trk_path, all_trks, reset=True)

    all_trks = flag_entryway_events(all_trks, entryways)
    all_trks = filter_entryway_stays(all_trks, entryways)

    frame_span = metadata['frame_span']
    all_frames = list(range(frame_span[0], frame_span[1] + 1))
    
    headcounts = global_headcounts(all_frames, all_trks)
    sections = headcount_sections(all_trks, headcounts)
    
    # Attempt to identify entry tracks and perform logical associations
    entry_subset = [trk for trk in all_trks.keys() if all_trks[trk]['entry']]
    print(entry_subset)
    all_trks = identify_tracks(all_trks, entry_subset, min_span=240)

    for section in sorted(sections.keys()):
        headcount = headcounts[sections[section]['span'][0]]
        all_trks = logical_identification(all_trks, sections[section], headcount)

    # Attempt to identify exit tracks and perform logical associations
    exit_subset = [trk for trk in all_trks.keys() if all_trks[trk]['exit']
                   and not all_trks[trk].get('identity', None)]
    print(exit_subset)
    all_trks = identify_tracks(all_trks, exit_subset, min_span=240)

    for section in sorted(sections.keys()):
        headcount = headcounts[sections[section]['span'][0]]
        all_trks = logical_identification(all_trks, sections[section], headcount)
    
    # Find biggest unidentified tracks
    for trk, data in all_trks.items():
        avg = (sum([box[2] * box[3] for box in data['detections'].values()])
               / len(data['detections'].keys()))
        data['box_avg'] = int(round(avg, 0))
    
    large_box_subset = []
    for trk in sorted(all_trks.keys(), key=lambda k: all_trks[k]['box_avg'],
                      reverse=True):
        if trk in entry_subset + exit_subset:
                continue
        data = all_trks[trk]
        if not data.get('identity', False):
            large_box_subset.append(trk)
            if len(large_box_subset) >= 4:
                break
    all_trks = identify_tracks(all_trks, large_box_subset)
    for section in sorted(sections.keys()):
        headcount = headcounts[sections[section]['span'][0]]
        all_trks = logical_identification(all_trks, sections[section], headcount)
    
    # Attempt to identify secondary tracks
    secondary_subset = []
    for k, v in all_trks.items():
        if ((k.split('_')[0] == primary_cams[0]) and (v['exit'] or v['entry'])
            and (not v.get('identity', None))):
            for k2, v2 in secondary_trks.items():
                if is_coincident(v['trk_span'], v2['trk_span']):
                    secondary_subset.append(k2)
    for k in secondary_subset:
        all_trks[k] = secondary_trks[k]    
    all_trks = identify_tracks(all_trks, secondary_subset, min_span=210)
    for trk in secondary_subset:
        if not all_trks[trk].get('identity', False):
            del all_trks[trk]
    sections = headcount_sections(all_trks, headcounts)

    for section in sorted(sections.keys()):
        headcount = headcounts[sections[section]['span'][0]]
        all_trks = logical_identification(all_trks, sections[section], headcount)
    

    # Attempt to identify remaining tracks
    for trk, data in all_trks.items():
        if not data.get('box_avg', False):
            avg = (sum([box[2] * box[3] for box in data['detections'].values()])
                / len(data['detections'].keys()))
            data['box_avg'] = int(round(avg, 0))

    final_subset = []
    for trk in sorted(all_trks.keys(), key=lambda k: all_trks[k]['box_avg'],
                      reverse=True):
        if trk in entry_subset + exit_subset + secondary_subset:
                continue
        data = all_trks[trk]
        if not data.get('identity', False):
            final_subset.append(trk)
    all_trks = identify_tracks(all_trks, final_subset)
    for section in sorted(sections.keys()):
        headcount = headcounts[sections[section]['span'][0]]
        all_trks = logical_identification(all_trks, sections[section], headcount)

    # Print results
    end = datetime.datetime.now()

    io_utils.write_track_ids(location, timestamp, all_trks)
    io_utils.write_trackspans(location, timestamp, all_frames, headcounts,
                              all_trks, granularity=90)

    io_utils.update_identities(trk_path, all_trks)
    
    print(f'main execution time: {(end - start).total_seconds()}')
