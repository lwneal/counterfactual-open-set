import math
import os
import tempfile
import time
import subprocess
import pathlib
from distutils import spawn
import numpy as np
from PIL import Image, ImageFont, ImageDraw
from io import BytesIO

# Should be available on Ubuntu 14.04+
#FONT_FILE = '/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf'
# Well apparently not. Hey, here's one installed with CUDA:
FONT_FILE = '/usr/local/cuda/jre/lib/fonts/LucidaSansRegular.ttf'

# Input: Numpy array containing one or more images
# Output: JPG encoded image bytes
def encode_jpg(pixels, resize_to=None):
    while len(pixels.shape) > 3:
        pixels = combine_images(pixels)
    # Convert to RGB to avoid "Cannot handle this data type"
    if pixels.shape[-1] < 3:
        pixels = np.repeat(pixels, 3, axis=-1)
    img = Image.fromarray(pixels.astype(np.uint8))
    if resize_to:
        img = img.resize(resize_to)
    fp = BytesIO()
    img.save(fp, format='JPEG')
    return fp.getvalue()


# Input: Filename, or JPG bytes
# Output: Numpy array containing images
def decode_jpg(jpg, crop_to_box=None, resize_to=(224,224), pil=False):
    if jpg.startswith('\xFF\xD8'):
        # Input is a JPG buffer
        img = Image.open(BytesIO(jpg))
    else:
        # Input is a filename
        img = Image.open(jpg)

    img = img.convert('RGB')
    if crop_to_box:
        # Crop to bounding box
        x0, x1, y0, y1 = crop_to_box
        width, height = img.size
        absolute_box = (x0 * width, y0 * height, x1 * width, y1 * height)
        img = img.crop((int(i) for i in absolute_box))
    if resize_to:
        img = img.resize(resize_to)
    if pil:
        return img
    return np.array(img).astype(float)

figure = []
def add_to_figure(data):
    figure.append(data)

def show_figure(**kwargs):
    global figure
    show(np.array(figure), **kwargs)
    figure = []

# Swiss-army knife for putting an image on the screen
# Accepts numpy arrays, PIL Image objects, or jpgs
# Numpy arrays can consist of multiple images, which will be collated
def show(
        data,
        verbose=False,
        display=True,
        save=True,
        filename=None,
        box=None,
        video_filename=None,
        resize_to=None,
        normalize_color=True,
        caption=None,
        font_size=16):
    # Munge data to allow input filenames, pixels, PIL images, etc
    if type(data) == type(np.array([])):
        pixels = data
    elif type(data) == Image.Image:
        pixels = np.array(data)
    elif type(data).__name__ == 'FloatTensor':
        # Pytorch tensor
        pixels = data.cpu().numpy()
        if len(pixels.shape) == 4:
            pixels = pixels.transpose((0,2,3,1))
        elif len(pixels.shape) == 3 and pixels.shape[0] in (1, 3):
            pixels = pixels.transpose((1,2,0))
    elif type(data).__name__ == 'Variable':
        # Pytorch tensor container
        pixels = data.data.cpu().numpy()
        if len(pixels.shape) == 4:
            pixels = pixels.transpose((0,2,3,1))
        elif len(pixels.shape) == 3 and pixels.shape[0] in (1, 3):
            pixels = pixels.transpose((1,2,0))
    elif type(data).__name__ == 'AxesSubplot':
        data.get_figure().savefig('/tmp/plot.jpg')
        pixels = np.array(Image.open('/tmp/plot.jpg'))
    elif hasattr(data, 'startswith'):
        pixels = decode_jpg(data, resize_to=resize_to)
    else:
        pixels = np.array(data)

    # Split non-RGB images into sets of monochrome images
    if pixels.shape[-1] not in (1, 3):
        pixels = np.expand_dims(pixels, axis=-1)

    # Expand matrices to monochrome images
    while len(pixels.shape) < 3:
        pixels = np.expand_dims(pixels, axis=-1)

    # Reduce lists of images to a single image
    while len(pixels.shape) > 3:
        pixels = combine_images(pixels)

    # Normalize pixel intensities
    if normalize_color and pixels.max() > pixels.min():
        pixels = (pixels - pixels.min()) * 255. / (pixels.max() - pixels.min())

    # Resize image to desired shape
    if resize_to:
        if pixels.shape[-1] == 1:
            pixels = pixels.repeat(3, axis=-1)
        img = Image.fromarray(pixels.astype('uint8'))
        img = img.resize(resize_to)
        pixels = np.array(img)

    # Draw a bounding box onto the image
    if box is not None:
        draw_box(pixels, box)

    # Draw text into the image
    if caption is not None:
        img = Image.fromarray(pixels.astype('uint8'))
        font = ImageFont.truetype(FONT_FILE, font_size)
        draw = ImageDraw.Draw(img)
        textsize = draw.textsize(caption, font=font)
        draw.rectangle([(0, 0), textsize], fill=(0,0,0,128))
        draw.multiline_text((0,0), caption, font=font, fill=(255,255,255))
        pixels = np.array(img)

    # Set a default filename if one does not exist
    if save and filename is None and video_filename is None:
        filename = '{}.jpg'.format(int(time.time() * 1000))
    elif filename is None:
        filename = tempfile.NamedTemporaryFile(suffix='.jpg').name

    # Write the file itself
    ensure_directory_exists(filename)
    with open(filename, 'wb') as fp:
        fp.write(encode_jpg(pixels))
        fp.flush()

    should_show = os.environ.get('IMUTIL_SHOW') and len(os.environ['IMUTIL_SHOW']) > 0 and spawn.find_executable('imgcat')
    if display and should_show:
        print('\n' * 4)
        print('\033[4F')
        subprocess.check_call(['imgcat', filename])
        print('\033[4B')
    elif verbose:
        print("Saved image size {} as {}".format(pixels.shape, filename))

    # Output JPG files can be collected into a video with ffmpeg -i *.jpg
    if video_filename:
        ensure_directory_exists(video_filename)
        open(video_filename, 'ab').write(encode_jpg(pixels))


def ensure_directory_exists(filename):
    # Assume whatever comes after the last / is the filename
    tokens = filename.split('/')[:-1]
    # Perform a mkdir -p on the rest of the path
    path = '/'.join(tokens)
    pathlib.Path(path).mkdir(parents=True, exist_ok=True)


def encode_video(video_filename):
    output_filename = video_filename.replace('mjpeg', 'mp4')
    print('Encoding video {}'.format(video_filename))
    cmd = 'ffmpeg -hide_banner -nostdin -loglevel warning -y -i {0} {1} && rm {0}'.format(video_filename, output_filename)
    os.system(cmd)


def combine_images(generated_images):
    num = generated_images.shape[0]
    width = int(math.sqrt(num))
    height = int(math.ceil(float(num)/width))
    shape = generated_images.shape[1:]
    image = np.zeros((height*shape[0], width*shape[1], shape[2]), dtype=generated_images.dtype)
    for index, img in enumerate(generated_images):
        i = int(index/width)
        j = index % width
        a0, a1 = i*shape[0], (i+1)*shape[0]
        b0, b1 = j*shape[1], (j+1)*shape[1]
        image[a0:a1, b0:b1] = img
    return image


def draw_box(img, box, color=1.0):
    height, width, channels = img.shape
    if all(0 < i < 1.0 for i in box):
        box = np.multiply(box, (width, width, height, height))
    x0, x1, y0, y1 = (int(val) for val in box)
    x0 = np.clip(x0, 0, width-1)
    x1 = np.clip(x1, 0, width-1)
    y0 = np.clip(y0, 0, height-1)
    y1 = np.clip(y1, 0, height-1)
    img[y0:y1,x0] = color
    img[y0:y1,x1] = color
    img[y0,x0:x1] = color
    img[y1,x0:x1] = color


class VideoMaker(object):
    def __init__(self, filename):
        self.filename = filename

    def write_frame(self, frame, caption=None):
        show(frame * 255.0,
                video_filename=self.filename,
                caption=caption,
                font_size=12,
                resize_to=(512,512),
                normalize_color=False,
                display=False)

    def finish(self):
        encode_video(self.filename)

