import os
import time


LOCK_FILE = 'worker_lockfile'


# TODO: Use a more sophisticated atomic lock
def acquire_lock(result_dir):
    filename = os.path.join(result_dir, LOCK_FILE)
    if os.path.exists(filename):
        raise OSError("{} is locked by another process".format(filename))
    with open(filename, 'w') as fp:
        fp.write(str(int(time.time())))


def release_lock(result_dir):
    filename = os.path.join(result_dir, LOCK_FILE)
    os.remove(filename)
