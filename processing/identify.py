import numpy as np
import bisect
import io_utils
import utilities
import subprocess
import sys
import tensorflow as tf
import datetime
import cv2
from deepface import DeepFace
import os


def identification_pipeline(time_prefix, min_span=120):
    def _setup(time_prefix):
        physical_devices = tf.config.list_physical_devices('GPU')
        if physical_devices:
            try:
                tf.config.experimental.set_visible_devices(physical_devices[0], 'GPU')
                print("Using GPU: ", physical_devices[0], flush=True)
            except RuntimeError as e:
                tf.config.experimental.set_visible_devices([], 'GPU')
                print(e, flush=True)
        else:
            tf.config.experimental.set_visible_devices([], 'GPU')
            print("No GPU available", flush=True)

        base_path = f'../intermediate_output/{time_prefix}'
        trk_path = base_path + '_trk_data.hdf5'

        config = io_utils.get_config()
        primary_cams = config['primary_cameras']
        entryways = config['entryways']

        metadata, all_trks = io_utils.get_trk_data(trk_path, primary_cams, min_span=min_span)
        io_utils.update_identities(trk_path, all_trks, reset=True)

        frame_span = metadata['frame_span']
        all_frames = list(range(frame_span[0], frame_span[1] + 1))

        return all_trks, all_frames, entryways

    def _handle_events(all_trks, time_prefix, entryways):
        def _filter_entryway_stay_events(all_trks, entryways, stride=6, threshold=.3):
            remove = []
            for id, trk in all_trks.items():
                if (not trk['entry']) or (not trk['exit']):
                    continue
                cam = id.split('_')[0]

                detections = trk['detections']
                frames = sorted(detections.keys())
                
                start = frames[0]
                for points in entryways[cam].values():
                    pcnt_in_entryway = utilities.percent_in_polygon(detections[start], points)
                    if pcnt_in_entryway > threshold:
                        break

                for f in frames:
                    if (f - start) % stride == 0:
                        pcnt_in_entryway = utilities.percent_in_polygon(detections[f], points)
                        if pcnt_in_entryway < threshold:
                            break
                    elif f == frames[-1]:
                        remove.append(id) 
            for id in remove:
                del all_trks[id]
            return all_trks

        all_trks = utilities.flag_entryway_events(all_trks, entryways,
                                                  threshold=.4)
        all_trks = _filter_entryway_stay_events(all_trks, entryways)

        io_utils.save_track_info(time_prefix, all_trks)

        return all_trks
    
    def _expand_scene_data(all_trks, all_frames):
        def _global_headcounts(all_trks, all_frames, prev_headcount=0):
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

        def _headcount_sections(all_trks, headcounts):
            section, sections = 0, {}
            frames = sorted(headcounts.keys())
            prev_count = headcounts[frames[0]]
            sections[section] = {'span': [frames[0]], 'tracks': [],
                                'headcount': prev_count}
            for frame in frames:
                count = headcounts[frame]
                if count != prev_count:
                    prev_count = count
                    sections[section]['span'].append(frame - 1)
                    section += 1
                    sections[section] = {'span': [frame], 'tracks': [],
                                        'headcount': count}

            sections[section]['span'].append(frames[-1])

            for section in sections.keys():
                for trk, data in all_trks.items():
                    if utilities.is_coincident(sections[section]['span'], data['trk_span']):
                        sections[section]['tracks'].append(trk)
            return sections
        
        def _calculate_avg_box_sizes(all_trks):
            for trk, data in all_trks.items():
                avg = (sum([box[2] * box[3] for box in data['detections'].values()])
                    / len(data['detections'].keys()))
                data['box_avg'] = int(round(avg, 0))
            return all_trks

        prev_headcount = io_utils.get_prev_headcount(time_prefix)
        if prev_headcount:
            headcounts = _global_headcounts(all_trks, all_frames, prev_headcount=prev_headcount)
        else:
            headcounts = _global_headcounts(all_trks, all_frames)
        sections = _headcount_sections(all_trks, headcounts)
        print(sections)
        io_utils.save_clip_headcounts(time_prefix, sections)

        all_trks = _calculate_avg_box_sizes(all_trks)

        return all_trks, headcounts, sections

    def _process_subset(time_prefix, all_trks, subset, sections, headcounts,
                    min_span=min_span):
        def _direct_identification(time_prefix, all_trks, subset, min_span):
            trks = [trk for trk in subset if (all_trks[trk]['trk_span'][1]
                                            - all_trks[trk]['trk_span'][0])
                                            >= min_span]
            for trk in trks:
                cam = trk.split('_')[0].strip('c')
                vid_file = f'{time_prefix}_{cam}.mp4'
                
                img_match, distance, event_img = facial_recognition(vid_file, all_trks[trk])
                if img_match:
                    identity = io_utils.get_employee(img_match)
                    all_trks[trk]['identity'] = identity
                    all_trks[trk]['id_method'] = 'face'
                    all_trks[trk]['id_cost'] = float(distance)
                    all_trks[trk]['id_img'] = event_img

                    print(f'{trk} direct ID: {identity}')
            return all_trks

        def _indirect_identification(all_trks, section, headcount):
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
                return utilities.is_coincident(all_trks[trk1]['trk_span'],
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
                        all_trks[trk]['id_method'] = 'one_possibility'
                        print(f'{trk} ID by one possibility: {all_trks[link]["identity"]}')
                    del possibilities[trk]

                    for trk2, data in possibilities.items():
                        if (trk2 != trk) and (link in data) and (len(data) != 1):
                            idx = data.index(link)
                            del data[idx]
                one_possibility = [k for k, v in possibilities.items() if len(v) == 1]
            # print(possibilities)

            candidates = {}
            for identity, trks in identities.items():
                for k, v in possibilities.items():
                    if set(v) == set(v + trks):
                        candidates.setdefault(identity, []).append(k)

            newly_identified = []
            for identity, trks in candidates.items():
                for trk in trks:
                    i = trks.index(trk)
                    filtered = [x for i2, x in enumerate(trks) if i2 != i]
                    if set(possibilities[trk]) == set(possibilities[trk] + filtered):
                        newly_identified.append(trk)
                        all_trks[trk]['identity'] = identity
                        all_trks[trk]['id_method'] = 'elimination'
                        print(f'{trk} ID by elimination: {identity}')
            
            
            for identity, trks in candidates.items():
                candidates[identity] = [trk for trk in trks if trk not in newly_identified]
            for identity, trks in candidates.items():
                newly_identified = []
                for trk in trks:
                    i = trks.index(trk)
                    filtered = [x for i2, x in enumerate(trks) if i2 != i]
                    if set(possibilities[trk]) == set(possibilities[trk] + filtered):
                        newly_identified.append(trk)
                        all_trks[trk]['identity'] = identity
                        all_trks[trk]['id_method'] = 'elimination'
                        print(f'{trk} ID by elimination: {id}')

            return all_trks

        all_trks = _direct_identification(time_prefix, all_trks, subset, min_span)
    
        for section in sorted(sections.keys()):
            headcount = headcounts[sections[section]['span'][0]]
            all_trks = _indirect_identification(all_trks, sections[section], headcount)

        return all_trks

    def _largest_remaining_tracks(all_trks, prior_subsets, limit=4):
        large_box_subset = []
        for trk in sorted(all_trks.keys(), key=lambda k:
                          all_trks[k]['box_avg'], reverse=True):
            if trk in prior_subsets:
                continue
            data = all_trks[trk]
            if not data.get('identity', False):
                large_box_subset.append(trk)
                if len(large_box_subset) >= limit:
                    break
        return large_box_subset
    
    def _get_track_images(time_prefix, all_trks, vid_dir='../input_files/'):
        for id, data in all_trks.items():
            camera = id.split('_')[0].strip('c')
            vid_path = os.path.join(vid_dir, f'{time_prefix}_{camera}.mp4')
            frames = sorted(data['detections'].keys())

            cap = cv2.VideoCapture(vid_path)
            if not cap.isOpened():
                return None
            
            images = []
            for f in [frames[0], frames[-1]]:
                x, y, w, h = data['detections'][f][:4]
                cap.set(cv2.CAP_PROP_POS_FRAMES, f)
                ret, frame = cap.read()
                if not ret:
                    continue
                cropped = frame[y:y+h, x:x+w]
                images.append(cropped)

            data['start_img'] = io_utils.save_event_image(images[0])
            data['end_img'] = io_utils.save_event_image(images[1])

        cap.release()
        return all_trks

    all_trks, all_frames, entryways = _setup(time_prefix)
    
    all_trks = _handle_events(all_trks, time_prefix, entryways)

    all_trks, headcounts, sections = _expand_scene_data(all_trks, all_frames)
    
    entry_subset = [trk for trk in all_trks.keys() if all_trks[trk].get('entry', False)]
    all_trks = _process_subset(time_prefix, all_trks, entry_subset, sections,
                            headcounts, min_span=min_span)

    exit_subset = [trk for trk in all_trks.keys() if all_trks[trk].get('exit', False)
                   and not all_trks[trk].get('identity', None)
                   and trk not in entry_subset]
    all_trks = _process_subset(time_prefix, all_trks, exit_subset, sections,
                               headcounts, min_span=min_span)

    prior_subsets = entry_subset + exit_subset
    large_box_subset = _largest_remaining_tracks(all_trks, prior_subsets)
    all_trks = _process_subset(time_prefix, all_trks, large_box_subset,
                               sections, headcounts, min_span=min_span)

    all_trks = _get_track_images(time_prefix, all_trks)

    updates = {}
    for id, data in all_trks.items():
        identity = data.get('identity', None)
        id_method = data.get('id_method', '')
        id_cost = data.get('id_cost', float('inf'))

        start_img = data.get('start_img', '')
        end_img = data.get('end_img', '')
        id_img = data.get('id_img', '')

        entry = 1 if data.get('entry', False) else 0
        exit = 1 if data.get('exit', False) else 0
        
        updates[id] = {'time_prefix': time_prefix, 'identity': identity,
                           'id_method': id_method, 'id_cost': id_cost,
                           'start_img': start_img, 'end_img': end_img,
                           'id_img': id_img, 'entry': entry,
                           'exit': exit}

    io_utils.update_track_info(time_prefix, updates)


