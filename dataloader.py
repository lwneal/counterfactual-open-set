import math
import time
import torch
from dataset_file import DatasetFile
from converter import ImageConverter, LabelConverter, FlexibleLabelConverter
from torchvision import transforms


class CustomDataloader(object):
    def __init__(self, dataset='mnist.dataset', batch_size=16, fold='train', shuffle=True, last_batch=False, example_count=None, **kwargs):
        self.dsf = DatasetFile(dataset, example_count=example_count)
        self.img_conv = ImageConverter(self.dsf, **kwargs)
        self.lab_conv = LabelConverter(self.dsf, **kwargs)
        self.batch_size = batch_size
        self.fold = fold
        self.last_batch = last_batch
        self.shuffle = shuffle
        self.num_classes = self.lab_conv.num_classes
        self.image_tensor = None
        self.label_tensor = None

    def get_batch(self, **kwargs):
        batch = self.dsf.get_batch(fold=self.fold, batch_size=self.batch_size, **kwargs)
        images, labels = self.convert(batch)
        return images, labels

    def __iter__(self):
        batcher = self.dsf.get_all_batches(
                fold=self.fold,
                batch_size=self.batch_size,
                shuffle=self.shuffle,
                last_batch=self.last_batch)
        for batch in batcher:
            images, labels = self.convert(batch)
            yield images, labels

    def convert(self, batch):
        images = self.img_conv(batch)
        labels = self.lab_conv(batch)
        images = torch.FloatTensor(images).cuda()
        labels = torch.LongTensor(labels).cuda()
        return images, labels

    def __len__(self):
        return math.floor(self.dsf.count(self.fold) / self.batch_size)

    def count(self):
        return self.dsf.count(self.fold)

    def class_name(self, idx):
        return lab_conv.labels[idx]


class FlexibleCustomDataloader(CustomDataloader):
    def __init__(self, dataset='mnist.dataset', batch_size=16, fold='train', shuffle=True, last_batch=False, example_count=None, **kwargs):
        super().__init__(dataset, batch_size, fold, shuffle, last_batch, example_count, **kwargs)
        self.lab_conv = FlexibleLabelConverter(dataset=self.dsf, **kwargs)

    def convert(self, batch):
        images = self.img_conv(batch)
        labels = self.lab_conv(batch)
        images = torch.FloatTensor(images).cuda()
        labels = torch.FloatTensor(labels).cuda()
        return images, labels
