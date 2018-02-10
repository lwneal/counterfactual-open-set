import os
import subprocess
import shutil


def mkdirp(path):
    os.makedirs(path, exist_ok=True)


def copy_repo(target_dir):
    # Creates a snapshot copy of this repository in target_dir
    # The copy should include any uncommitted or unstaged changes
    # (This makes experiments reproducible, even if they're not checked in)

    # Get the list of tracked filenames in this repo
    stdout = subprocess.check_output(['git', 'ls-files'])
    filenames = str(stdout, 'utf-8').splitlines()
    for src_filename in filenames:
        dst_filename = os.path.join(target_dir, src_filename)
        mkdirp(os.path.dirname(dst_filename))
        shutil.copy2(src_filename, dst_filename)
    print('Copied {} files to {}'.format(len(filenames), target_dir))

