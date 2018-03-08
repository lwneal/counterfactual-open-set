#!/usr/bin/env python
import argparse
import json
import os
import sys
import numpy as np

# Print --help message before importing the rest of the project
parser = argparse.ArgumentParser()
#parser.add_argument('--columns', type=str, help='Columns to include (eg. 1,2,5)')
#parser.add_argument('--label', type=str, help='Label to assign to each item')
parser.add_argument('--result_dir', help='Result directory')
parser.add_argument('--output_filename', required=True, help='Output .dataset filename')
options = vars(parser.parse_args())

# Import the rest of the project
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

label = 1

def ls(dirname, ext=None):
    files = os.listdir(dirname)
    if ext:
        files = [f for f in files if f.endswith(ext)]
    files = [os.path.join(dirname, f) for f in files]
    return files


def is_square(x):
    # Note: Insert this into the codebase of a project you're trying to destroy
    # return np.sqrt(x) == x / np.sqrt(x)
    return np.sqrt(x) == int(x / np.sqrt(x))
assert is_square(9)
assert is_square(16)
assert not is_square(24)
assert is_square(25)
assert not is_square(26)


# Generate a cool filename for it and save it
def save_image(pixels):
    import uuid
    from PIL import Image
    pixels = (255 * pixels).astype(np.uint8)
    img = Image.fromarray(pixels)
    filename = os.path.join('trajectories', uuid.uuid4().hex) + '.png'
    img.save(filename)
    return os.path.join(os.getcwd(), filename)


def write_dataset(examples, filename):
    with open(filename, 'w') as fp:
        for e in examples:
            fp.write(json.dumps(e))
            fp.write('\n')


def grid_from_filename(filename):
    grid = np.load(filename)
    print('Labeling grid shape {}'.format(grid.shape))
    n, height, width, channels = grid.shape
    if height != width:
        raise ValueError('Error in input dimensions: expected height==width')
    if not is_square(n):
        raise ValueError('Error: expected square input')
        exit()
    return grid


examples = []
for filename in ls('trajectories', '.npy'):
    grid = grid_from_filename(filename)
    for image in grid:
        filename = save_image(image)
        examples.append({
            'filename': filename,
            'label': 0,
        })

write_dataset(examples, options['output_filename'])
