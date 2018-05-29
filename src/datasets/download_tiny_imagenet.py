#!/usr/bin/env python
import os
import random
import numpy as np
import json
from subprocess import check_output


DATA_ROOT_DIR = '/mnt/nfs/data'
DATASET_DIR = os.path.join(DATA_ROOT_DIR, 'tiny_imagenet')
DATASET_NAME = 'tiny_imagenet'

ANIMAL_CLASSES = [
    "dragonfly, darning needle, devil's darning needle, sewing needle, snake feeder, snake doctor, mosquito hawk, skeeter hawk",
    'African elephant, Loxodonta africana',
    'American alligator, Alligator mississipiensis',
    'American lobster, Northern lobster, Maine lobster, Homarus americanus',
    'Arabian camel, dromedary, Camelus dromedarius',
    'Chihuahua',
    'Egyptian cat',
    'European fire salamander, Salamandra salamandra',
    'German shepherd, German shepherd dog, German police dog, alsatian',
    'Labrador retriever',
    'Persian cat',
    'Yorkshire terrier',
    'albatross, mollymawk',
    'baboon',
    'bee',
    'bighorn, bighorn sheep, cimarron, Rocky Mountain bighorn, Rocky Mountain sheep, Ovis canadensis',
    'bison',
    'black stork, Ciconia nigra',
    'black widow, Latrodectus mactans',
    'boa constrictor, Constrictor constrictor',
    'brown bear, bruin, Ursus arctos',
    'bullfrog, Rana catesbeiana',
    'centipede',
    'chimpanzee, chimp, Pan troglodytes',
    'cockroach, roach',
    'cougar, puma, catamount, mountain lion, painter, panther, Felis concolor',
    'dugong, Dugong dugon',
    'feeder, snake doctor, mosquito hawk, skeeter hawk',
    'fly',
    'gazelle',
    'golden retriever',
    'goldfish, Carassius auratus',
    'goose',
    'grasshopper, hopper',
    'guinea pig, Cavia cobaya',
    'hog, pig, grunter, squealer, Sus scrofa',
    'jellyfish',
    'king penguin, Aptenodytes patagonica',
    'koala, koala bear, kangaroo bear, native bear, Phascolarctos cinereus',
    'ladybug, ladybeetle, lady beetle, ladybird, ladybird beetle',
    'lesser panda, red panda, panda, bear cat, cat bear, Ailurus fulgens',
    'lion, king of beasts, Panthera leo',
    'mantis, mantid',
    'monarch, monarch butterfly, milkweed butterfly, Danaus plexippus',
    'orangutan, orang, orangutang, Pongo pygmaeus',
    'ox',
    'scorpion',
    'sea cucumber, holothurian',
    'sea slug, nudibranch',
    'sheep, Ovis canadensis',
    'slug',
    'snail',
    'spiny lobster, langouste, rock lobster, crawfish, crayfish, sea crawfish',
    'standard poodle',
    'sulphur butterfly, sulfur butterfly',
    'tabby, tabby cat',
    'tailed frog, bell toad, ribbed toad, tailed toad, Ascaphus trui',
    'tarantula',
    'trilobite',
    'walking stick, walkingstick, stick insect',
]



def mkdir(path):
    path = os.path.expanduser(path)
    if not os.path.exists(path):
        print('Creating directory {}'.format(path))
        os.mkdir(path)


def download(filename, url):
    if os.path.exists(filename):
        print("File {} already exists, skipping".format(filename))
    else:
        os.system('wget -nc {}'.format(url))
        if url.endswith('.tgz') or url.endswith('.tar.gz'):
            os.system('ls *gz | xargs -n 1 tar xzvf')


def get_width_height(filename):
    from PIL import Image
    img = Image.open(os.path.expanduser(filename))
    return (img.width, img.height)


def save_dataset(examples, output_filename):
    print("Writing {} items to {}".format(len(examples), output_filename))
    fp = open(output_filename, 'w')
    for example in examples:
        fp.write(json.dumps(example) + '\n')
    fp.close()


if __name__ == '__main__':
    print("Downloading dataset {}...".format(DATASET_NAME))
    mkdir(DATA_ROOT_DIR)
    mkdir(DATASET_DIR)
    os.chdir(DATASET_DIR)

    # Download and extract dataset
    print("Downloading dataset files...")
    download('tiny-imagenet-200', 'http://cs231n.stanford.edu/tiny-imagenet-200.zip')

    # Remove extra directory
    os.system('mv tiny-imagenet-200/* . && rmdir tiny-imagenet-200')

    wnids = open('wnids.txt').read().splitlines()

    wnid_names = {}
    for line in open('words.txt').readlines():
        wnid, name = line.strip().split('\t')
        wnid_names[wnid] = name

    test_filenames = os.listdir('test/images')

    examples = []

    # Collect training examples
    for wnid in os.listdir('train'):
        filenames = os.listdir(os.path.join('train', wnid, 'images'))
        for filename in filenames:
            file_path = os.path.join(DATASET_NAME, 'train', wnid, 'images', filename)
            examples.append({
                'filename': file_path,
                'label': wnid_names[wnid],
                'fold': 'train',
            })

    # Use validation set as a test set
    for line in open('val/val_annotations.txt').readlines():
        jpg_name, wnid, x0, y0, x1, y1 = line.split()
        examples.append({
            'filename': os.path.join(DATASET_NAME, 'val', 'images', jpg_name),
            'label': wnid_names[wnid],
            'fold': 'test',
        })
    save_dataset(examples, '{}/{}.dataset'.format(DATA_ROOT_DIR, DATASET_NAME))

    # Split animal and non/animal (plants, vehicles, objects, etc)
    animal_examples = [e for e in examples if e['label'] in ANIMAL_CLASSES]
    save_dataset(animal_examples, '{}/{}-animals.dataset'.format(DATA_ROOT_DIR, DATASET_NAME))

    not_animal_examples = [e for e in examples if e['label'] not in ANIMAL_CLASSES]
    save_dataset(not_animal_examples, '{}/{}-not-animals.dataset'.format(DATA_ROOT_DIR, DATASET_NAME))

    # Put the unlabeled test set in a separate dataset
    test_examples = []
    for jpg_name in os.listdir('test/images'):
        test_examples.append({
            'filename': os.path.join(DATASET_NAME, 'test', 'images', jpg_name),
            'fold': 'test',
        })
    save_dataset(test_examples, '{}/{}-unlabeled.dataset'.format(DATA_ROOT_DIR, DATASET_NAME))

    # Select a random 10, 50, 100 classes and partition them out
    classes = list(set(e['label'] for e in examples))

    random.seed(42)
    for known_classes in [10, 20, 50]:
        for i in range(5):
            random.shuffle(classes)
            known = [e for e in examples if e['label'] in classes[:known_classes]]
            unknown = [e for e in examples if e['label'] not in classes[:known_classes]]
            save_dataset(known, '{}/{}-known-{}-split{}a.dataset'.format(DATA_ROOT_DIR, DATASET_NAME, known_classes, i))
            save_dataset(unknown, '{}/{}-known-{}-split{}b.dataset'.format(DATA_ROOT_DIR, DATASET_NAME, known_classes, i))

    print("Finished building dataset {}".format(DATASET_NAME))
