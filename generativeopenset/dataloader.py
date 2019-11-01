import math
import torch
from dataset_file import DatasetFile
from converter import ImageConverter, LabelConverter, FlexibleLabelConverter


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
        """
        # TODO: Multithreading improves throughput by 10-20%
        # It must be implemented safely, however- not like this
        # In particular, ensure no deadlocks, interactivity and logging should still work
        import queue
        import threading
        q = queue.Queue(maxsize=1)
        def yield_batch_worker():
            for batch in batcher:
                images, labels = self.convert(batch)
                q.put((images, labels))
            q.put('finished')
        t = threading.Thread(target=yield_batch_worker)
        t.start()
        while True:
            result = q.get()
            if result == 'finished':
                break
            yield result
        t.join()
        """
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
        return self.lab_conv.labels[idx]


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
