# standard dependencies
import logging
from logging.handlers import RotatingFileHandler
from psutil import NoSuchProcess, AccessDenied, ZombieProcess
import tracemalloc
import os
import psutil
import time
import threading
from datetime import datetime
import gc
import sys

# 3rd-party dependencies
import torch

# internal dependencies
pass


def configure_logging():
    log_dir = os.path.abspath(os.path.join(os.getcwd(), '..', 'files', 'logs'))
    os.makedirs(log_dir, exist_ok=True)

    log_file = os.path.join(log_dir, 'app.log')
    MB = 1024 * 1024

    formatter = logging.Formatter(
        fmt="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )

    file_handler = RotatingFileHandler(
        log_file,
        maxBytes=(500 * MB),  # 500 MB per file
        backupCount=4         # Up to 2 GB total
    )
    file_handler.setFormatter(formatter)

    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s [%(levelname)s] %(name)s PID[%(process)d] %(message)s',
        handlers=[file_handler, stream_handler]
    )


configure_logging()


def get_logger(name=None):
    return logging.getLogger(name or __name__)


def press_stopwatch(instance, target_attr: str):
    accrued_time = getattr(instance, target_attr, 0)
    start_attr = f'start_{target_attr}'

    if not hasattr(instance, start_attr):
        start = time.perf_counter() # create start reference
        setattr(instance, start_attr, start)
        return True
    
    else:
        start = getattr(instance, start_attr)   # retrieve start reference
        stop = time.perf_counter()

        accrued_time += (stop - start)
        setattr(instance, target_attr, accrued_time)    # update total

        delattr(instance, start_attr)   # delete start reference
        return False


def observability_thread(target, args=None):
    """
    Initializes a thread for monitoring & logging some aspect of a process.

    Args:
        target (string): The aspect of the process to observe. This determines
            which function is assigned to the `target` parameter of the thread
            constructor. Options: 'elapsed_time', 'failed_workers', 'low_memory'

        args (tuple): Any arguments that the target function expects.
            - elapsed_time: (frequency=300, include_timestamp=False)
            - failed_workers: (pool, initial_pids, async_results)
            - low_memory: (threshold=1000, interval=1)
    """

    # To do: implement logger class with .stop() method to set stop events
    # rather than having to return the stop event separately

    if target == 'elapsed_time':
        start_time, stop_event = time.time(), threading.Event()
        frequency, timestamp = args or (300, False)
    
        time_logger = threading.Thread(
            target=log_elapsed_time,
            args=(start_time, stop_event, frequency, timestamp),
            daemon=True
        )
        return time_logger, stop_event

    elif target == 'failed_workers':
        worker_monitor = threading.Thread(
            target=log_failed_workers, args=args, daemon=True
        )
        return worker_monitor
    
    elif target == 'low_memory':
        stop_event = threading.Event()
        threshold, interval = args or (1000, 1)

        low_memory_monitor = threading.Thread(
            target=log_low_memory_warnings,
            args=(stop_event, threshold, interval),
            daemon=True
        )
        return low_memory_monitor, stop_event


def log_elapsed_time(start_time, stop_event, frequency, timestamp):
    while not stop_event.is_set():
        elapsed = (time.time() - start_time) / 60
        if timestamp == False:
            print(f'Elapsed time: {elapsed:.2f} minutes')
            time.sleep(frequency)
        elif timestamp == True:
            current_time = datetime.now().strftime('%H:%M:%S')
            print(f'[{current_time}] Elapsed time: {elapsed:.2f} minutes')
            time.sleep(frequency)

    total_elapsed =  (time.time() - start_time) / 60
    if timestamp == False:
        print(f'Total elapsed time: {total_elapsed:.2f} minutes')
    elif timestamp == True:
        current_time = datetime.now().strftime('%H:%M:%S')
        print(f'[{current_time}] Total elapsed time: {total_elapsed:.2f} minutes')


def log_failed_workers(pool, initial_pids, async_result):
    '''
    Logs any potentially failed workers from a starmap_async() run of a
    multiprocessing.Pool
    '''

    while not async_result.ready():
        time.sleep(1)
        current_pids = {p.pid for p in pool._pool if p.is_alive()}
        disappeared = initial_pids - current_pids
        if disappeared:
            print(f"[WARNING] Workers {disappeared} disappeared (possible crash)")


def memory_usage(focus, n=5, threshold=None, log_filter_key=None):
    if focus == 'processes':
        def _log_largest_processes(process_list, n):
            if process_list:
                print(f'Largest processes:')
                for pid, name, mem in processes[:n]:
                    print(f'PID {pid} - {name}: {mem:.2f} MB')

        processes = []
        for p in psutil.process_iter(
            attrs=['pid', 'name', 'memory_info'], ad_value=None
        ):
            try:
                info = p.as_dict(attrs=['pid', 'name', 'memory_info'])
                if info['memory_info']:
                    processes.append(
                        (info['pid'], info['name'], info['memory_info'].rss / 1e6)
                    )
            except (NoSuchProcess, AccessDenied, ZombieProcess):
                continue

        processes.sort(key=lambda x: x[2], reverse=True)

        total_process_memory = sum([process[2] for process in processes])
        if (threshold is None) or (total_process_memory > threshold):
            _log_largest_processes(processes, n)

        return total_process_memory

    elif focus == 'objects':
        def _log_largest_objects(object_list, n, obj_category):
            if object_list:
                print(f'Largest {obj_category} objects:')
                for obj, size in object_list[:n]:
                    print(f'Size: {size} MB | Type: {type(obj)}')

            else:
                print(f'No {obj_category} objects found')

        def _safe_sizeof(object):
            '''
            Returns the size of object in megabytes to two decimal places while
            safely handling exceptions.
            '''
            try:
                raw_size = sys.getsizeof(object)
                return round((raw_size / 1e6), 2)
            except TypeError:
                return 0

        gc.collect()

        standard_objects = sorted(
            [(obj, _safe_sizeof(obj)) for obj in gc.get_objects()],
            key=lambda x: x[1],
            reverse=True
        )
        uncollectible_objects = sorted(
            [(obj, _safe_sizeof(obj)) for obj in gc.garbage],
            key=lambda x: x[1],
            reverse=True
        )

        cpu_obj_totals = [sum([size for _, size in obj_list]) for obj_list in
                          [standard_objects, uncollectible_objects]]
        gpu_obj_totals = [(torch.cuda.memory_allocated() / 1e6)]
        
        total_obj_memory = sum(cpu_obj_totals) + sum(gpu_obj_totals)

        if (
            (threshold is None) or
            (total_obj_memory > (threshold))
        ):

            print(f'Total standard object memory: {cpu_obj_totals[0]:.2f} MB')
            _log_largest_objects(standard_objects, n, 'standard')
    
            print(f'Total uncollectible object memory: {cpu_obj_totals[1]:.2f} MB')
            _log_largest_objects(uncollectible_objects, n, 'uncollectible')

            print(f"Total pytorch object memory: {gpu_obj_totals[0]:.2f} MB")

        return total_obj_memory

    elif focus == 'allocation_lines':
        snapshot = tracemalloc.take_snapshot()
        allocation_lines = snapshot.statistics('lineno')

        allocation_lines = [(
            (line_info.traceback[-1].filename), (line_info.traceback[-1].lineno),
            (line_info.size / 1e6)
            ) for line_info in allocation_lines
        ]

        total_alloc_memory = sum([x[2] for x in allocation_lines])

        if log_filter_key is not None:
            allocation_lines = [
                (file, line_num, memory) for file, line_num, memory
                in allocation_lines if log_filter_key(file)
            ]
        if (
            (threshold is None) or
            (total_alloc_memory > threshold)
        ):
            print(f'Total allocated memory: {round(total_alloc_memory, 2)} MB')
            print('Top allocation lines:')
            for line_info in allocation_lines[:n]:
                file, line_num, memory = line_info

                print(
                    f'File {file}, line {line_num},' +
                    f'allocated {memory:.2f} MB'
                )

        return total_alloc_memory


def log_low_memory_warnings(stop_event, threshold, interval):
    while not stop_event.is_set():
        try:
            memory_info = psutil.virtual_memory()
            free_mb = memory_info.available / 1e6

            if free_mb < threshold:
                free_mb = round(free_mb, 0) if free_mb >= 1 else round(free_mb, 2)
                print(f'\n[WARNING] MEMORY CRITICAL: {free_mb} MB free')

                memory_usage('processes')
                memory_usage('objects')

                gc.collect()
                time.sleep(2)
                print('[CRITICAL] Exiting due to low memory')
                os._exit(1)
            else:
                time.sleep(interval)
        except Exception as e:
            print(f'Error while monitoring memory: {e}')


def dump_native_usage(tag='', logger=None):
    '''
    Displays the process's resident set size (rss), shared memory segments (shm),
    and pyTorch CPU usage.
    '''
    p = psutil.Process(os.getpid())
    rss = p.memory_info().rss / 1024**2     # MB
    shm = sum(m.rss for m in p.memory_maps()
              if '/dev/shm' in m.path) / 1024**2
    try:
        torch_cpu = torch._C._get_cpu_memory_usage() / 1024**2
    except AttributeError:
        torch_cpu = 0
    if logger:
        logger.info(f'{tag:20s} | rss {rss:,.0f} MB  shm {shm:,.0f} MB  torch-cpu {torch_cpu:,.0f} MB')
    else:
        print(f'{tag:20s} | rss {rss:,.0f} MB  shm {shm:,.0f} MB  torch-cpu {torch_cpu:,.0f} MB')
