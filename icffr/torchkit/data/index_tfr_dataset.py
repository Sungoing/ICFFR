from collections import defaultdict
import os
import struct
from io import BytesIO
from PIL import Image
from torch.utils.data import Dataset
from torchkit.data import example_pb2
import random
import numpy as np
import torch
import torch.nn.functional as F
import copy

def read_index_file(index_file):
    samples_offsets = []
    record_files = []
    labels = []
    with open(index_file, 'r') as ifs:
        for line in ifs:
            record_name, tf_record_index, tf_record_offset, label = line.rstrip().split('\t')
            samples_offsets.append(int(tf_record_offset))
            record_files.append(os.path.join(
                record_name,
                "%s-%05d.tfrecord" % (record_name, int(tf_record_index))
            ))
            labels.append(int(label))
    return record_files, samples_offsets, labels

def read_index_file_with_quality(index_file):
    samples_offsets = []
    record_files = []
    labels = []
    qualities = []
    with open(index_file, 'r') as ifs:
        for line in ifs:
            record_name, tf_record_index, tf_record_offset, label, quality = line.rstrip().split('\t')
            samples_offsets.append(int(tf_record_offset))
            record_files.append(os.path.join(
                record_name,
                "%s-%05d.tfrecord" % (record_name, int(tf_record_index))
            ))
            labels.append(int(label))
            qualities.append(float(quality))
    return record_files, samples_offsets, labels, qualities

def read_index_file_with_race_label(index_file):
    samples_offsets = []
    record_files = []
    labels = []
    race_labels = []
    with open(index_file, 'r') as ifs:
        for line in ifs:
            try:
                record_name, tf_record_index, tf_record_offset, label, race_label = line.rstrip().split('\t')
            except:
                print(line, ', length of line is: %d'%(len(line)))
            samples_offsets.append(int(tf_record_offset))
            record_files.append(os.path.join(
                record_name,
                "%s-%05d.tfrecord" % (record_name, int(tf_record_index))
            ))
            labels.append(int(label))
            if race_label == 'African':
                race_labels.append(0)
            elif race_label == 'Asian':
                race_labels.append(1)
            elif race_label == 'Caucasian':
                race_labels.append(2)
            elif race_label == 'Indian':
                race_labels.append(3)
            else:
                race_labels.append(int(race_label))
    return record_files, samples_offsets, labels, race_labels


def read_pair_index_file(index_file):
    samples_offsets = []
    record_files = []
    labels = []
    with open(index_file, 'r') as ifs:
        for line in ifs:
            (record_name_first, tf_record_index_first, tf_record_offset_first,
             label_first, record_name_second, tf_record_index_second,
             tf_record_offset_second, label_second) = line.rstrip().split('\t')
            samples_offsets.append((int(tf_record_offset_first), int(tf_record_offset_second)))
            record_files.append((os.path.join(
                record_name_first,
                "%s-%05d.tfrecord" % (record_name_first, int(tf_record_index_first))
            ), os.path.join(
                record_name_second,
                "%s-%05d.tfrecord" % (record_name_second, int(tf_record_index_second)))))
            labels.append((int(label_first), (label_second)))

    return (record_files, samples_offsets, labels)


class IndexTFRDataset(Dataset):
    """Index TFRecord Dataset
    """

    def __init__(self, tfrecord_dir, index_file, transform):
        """Create a ``IndexTFRDataset`` object
        A ``IndexTFRDataset`` object will read sample proto from *.tfrecord files saved
        in ``tfrecord_dir`` by index_file, the sample proto will convert to image and
        fed into Dataloader.

        Args:
            tfrecord_dir: tfrecord saved dir
            index_file: each line format ``tfr_name\t tfr_record_index \t tfr_record_offset \t label``
            transform: image transform
        """
        self.root_dir = tfrecord_dir
        self.index_file = index_file
        self.transform = transform
        self.records, self.offsets, self.labels = read_index_file(self.index_file)
        for record_file in set(self.records):
            record_file = os.path.join(self.root_dir, record_file)
            if not os.path.exists(record_file):
                raise RuntimeError("tfrecord file： %s not found" % record_file)
        self.class_num = max(self.labels) + 1
        self.sample_num = len(self.records)
        #self.sample_num = 128*300
        print('class_num: %d, sample_num:  %d' % (self.class_num, self.sample_num))

    def __len__(self):
        return self.sample_num

    def _parser(self, feature_list):
        for key, feature in feature_list:
            if key == 'image':
                image_raw = feature.bytes_list.value[0]
                image = Image.open(BytesIO(image_raw))
                image = image.convert('RGB')
                image = self.transform(image)
        return image

    def _get_record(self, record_file, offset):
        with open(record_file, 'rb') as ifs:
            ifs.seek(offset)
            byte_len_crc = ifs.read(12)
            proto_len = struct.unpack('Q', byte_len_crc[:8])[0]
            # proto,crc
            pb_data = ifs.read(proto_len)
            if len(pb_data) < proto_len:
                print("read pb_data err,proto_len:%s pb_data len:%s" % (proto_len, len(pb_data)))
                return None
        example = example_pb2.Example()
        example.ParseFromString(pb_data)
        # keep key value in order
        feature = sorted(example.features.feature.items())
        record = self._parser(feature)
        return record

    def __getitem__(self, index):
        offset = self.offsets[index]
        record = self.records[index]
        record_file = os.path.join(self.root_dir, record)
        return self._get_record(record_file, offset), self.labels[index]

class IndexTFRDatasetClassSample(IndexTFRDataset):
    """Index TFRecord by sampling classes
    """

    def __init__(self, tfrecord_dir, index_file, transform, balance_sample_times=False, efficient=True, repeat_sample=True,class_sample_num=8):
        
        self.root_dir = tfrecord_dir
        self.index_file = index_file
        self.transform = transform
        self.efficient = efficient
        self.repeat_sample = repeat_sample
        self.records, self.offsets, self.labels = read_index_file(self.index_file)
        for record_file in set(self.records):
            record_file = os.path.join(self.root_dir, record_file)
            if not os.path.exists(record_file):
                raise RuntimeError("tfrecord file： %s not found" % record_file)

        self.balance_sample_times = balance_sample_times
        self.class_num = max(self.labels) + 1
        self.sample_num = len(self.records)
        self.class_sample_num = class_sample_num
        self.class_records = self.get_class_records()
        self.class_sample_times = self.get_class_sample_times()
        self.train_class_list = self.get_train_class_list()
        self.remain_records = {}

    def get_class_tf(self):
        class_tf = defaultdict(list)
        for idx,label in enumerate(self.labels):
            class_tf[label].append(self.records[idx])
        return class_tf
    
    def get_class_offsets(self):
        class_offsets = defaultdict(list)
        for idx,label in enumerate(self.labels):
            class_offsets[label].append(self.offsets[idx])
        return class_offsets
    
    def get_class_records(self):
        class_records = defaultdict(list)
        class_offsets = self.get_class_offsets()
        class_tf = self.get_class_tf()
        for label in class_offsets.keys():
            for idx,offset in enumerate(class_offsets[label]):
                class_records[label].append((class_tf[label][idx], offset))
        return class_records

    def get_class_sample_times(self):
        sample_times = defaultdict(int)
        for label in self.labels:
            sample_num = len(self.class_records[label])

            class_sample_times = (sample_num + self.class_sample_num-1) // self.class_sample_num
            if self.repeat_sample and self.balance_sample_times and class_sample_times < 4:
                class_sample_times = 4
            sample_times[label] = class_sample_times
        return sample_times

    def get_train_class_list(self):
        train_class_list = []
        sample_times = self.class_sample_times
        for label in range(self.class_num):
            for i in range(sample_times[label]):
                train_class_list.append(label)
        random.shuffle(train_class_list)
        return train_class_list

    def get_class_random_records(self, label, required_num=8):
        required_samples = []
        required_labels = []
        random_class = label
        total_num = 0
        while total_num<required_num:
            key = str(random_class)
            if key not in self.remain_records:
                self.remain_records[key] = copy.deepcopy(self.class_records[random_class])
            
            records = self.class_records[random_class]
            
            if not self.repeat_sample:
                records = self.remain_records[key]
            
            random.shuffle(records)
            sample_num = len(records)
            accepted_num = min(required_num-total_num, sample_num)
            total_num += accepted_num
            required_samples += records[:accepted_num]
            for i in range(accepted_num):
                required_labels.append(random_class)
            
            self.remain_records[key] = self.remain_records[key][accepted_num:]
            if sample_num<required_num:
                del self.remain_records[key]
                if self.efficient:
                    for i in range(required_num - sample_num):
                        class_record_num = len(self.class_records[random_class])
                        random_idx = random.randint(0, class_record_num-1)
                        required_samples.append(self.class_records[random_class][random_idx])
                        required_labels.append(label)
                    total_num += required_num - sample_num
                else:
                    random_class = random.randint(0, self.class_num - 1)
        return required_samples, required_labels

    def __len__(self):
        return len(self.train_class_list)

    def __getitem__(self, index):
        img_list = []
        label_list = []
        class_records, records_labels = self.get_class_random_records(self.train_class_list[index], required_num=self.class_sample_num)
        for i,record_info in enumerate(class_records):
            record = record_info[0]
            record_file = os.path.join(self.root_dir, record)
            offset = record_info[1]
            img = self._get_record(record_file, offset)
            # print(img.shape)
            img = img.view(1, 3, 112, 112)
            img_list.append(img)
            label = records_labels[i]
            label_list.append(torch.Tensor([label]).long())
        return torch.cat(img_list, 0), torch.cat(label_list, 0)


class IndexTFRDatasetClassSampleWithRace(IndexTFRDataset):
    """Index TFRecord by sampling classes
    """

    def __init__(self, tfrecord_dir, index_file, transform):
        
        self.root_dir = tfrecord_dir
        self.index_file = index_file
        self.transform = transform
        self.records, self.offsets, self.labels, self.race_labels = read_index_file_with_race_label(self.index_file)
        for record_file in set(self.records):
            record_file = os.path.join(self.root_dir, record_file)
            if not os.path.exists(record_file):
                raise RuntimeError("tfrecord file： %s not found" % record_file)
        
        self.class_num = max(self.labels) + 1
        self.sample_num = len(self.records)
        self.race_num = np.max(self.race_labels) + 1

        self.class_records = self.get_class_records()
        self.class_sample_times = self.get_class_sample_times()
        self.train_class_list = self.get_train_class_list()

    def get_class_tf(self):
        class_tf = defaultdict(list)
        for idx,label in enumerate(self.labels):
            class_tf[label].append(self.records[idx])
        return class_tf
    
    def get_class_offsets(self):
        class_offsets = defaultdict(list)
        for idx,label in enumerate(self.labels):
            class_offsets[label].append(self.offsets[idx])
        return class_offsets
    
    def get_class_race(self):
        class_races = defaultdict(int)
        for idx,label in enumerate(self.labels):
            class_races[label] = self.race_labels[idx]
        return class_races

    def get_class_records(self):
        class_records = defaultdict(list)
        class_offsets = self.get_class_offsets()
        class_tf = self.get_class_tf()
        class_races = self.get_class_race()
        for label in class_offsets.keys():
            for idx,offset in enumerate(class_offsets[label]):
                class_records[label].append((class_tf[label][idx], offset, class_races[label]))
        return class_records

    def get_class_sample_times(self):
        sample_times = defaultdict(int)
        for label in self.labels:
            sample_num = len(self.class_records[label])
            class_sample_times = sample_num // 8 + 1
            sample_times[label] = class_sample_times
        return sample_times

    def get_train_class_list(self):
        train_class_list = []
        sample_times = self.class_sample_times
        for label in range(self.class_num):
            for i in range(sample_times[label]):
                train_class_list.append(label)
        random.shuffle(train_class_list)
        return train_class_list

    def get_class_random_records(self, label, required_num=8):
        required_samples = []
        required_labels = []
        random_class = label
        total_num = 0
        while total_num<required_num:
            records = self.class_records[random_class]
            random.shuffle(records)
            sample_num = len(records)
            accepted_num = min(required_num-total_num, sample_num)
            total_num += accepted_num
            required_samples += records[:accepted_num]
            for i in range(accepted_num):
                required_labels.append(random_class)
            random_class = random.randint(0, self.class_num - 1)
        return required_samples, required_labels

    def __len__(self):
        return len(self.train_class_list)

    def __getitem__(self, index):
        img_list = []
        label_list = []
        race_list = []
        class_records, records_labels = self.get_class_random_records(self.train_class_list[index], required_num=8)
        for i,record_info in enumerate(class_records):
            record = record_info[0]
            record_file = os.path.join(self.root_dir, record)
            offset = record_info[1]
            img = self._get_record(record_file, offset)
            # print(img.shape)
            img = img.view(1, 3, 112, 112)
            img_list.append(img)
            label = records_labels[i]
            label_list.append(torch.Tensor([label]).long())
            race_label = record_info[2]
            race_list.append(torch.Tensor([race_label]).long())
        return torch.cat(img_list, 0), torch.cat(label_list, 0), torch.cat(race_list, 0)


class IndexTFRSingleRace(IndexTFRDataset):
    def __init__(self, tfrecord_dir, index_file, transform):
        """Create a ``IndexTFRDataset`` object
        A ``IndexTFRDataset`` object will read sample proto from *.tfrecord files saved
        in ``tfrecord_dir`` by index_file, the sample proto will convert to image and
        fed into Dataloader.

        Args:
            tfrecord_dir: tfrecord saved dir
            index_file: each line format ``tfr_name\t tfr_record_index \t tfr_record_offset \t label``
            transform: image transform
        """
        self.root_dir = tfrecord_dir
        self.index_file = index_file
        self.transform = transform
        self.records, self.offsets, self.labels = read_index_file(self.index_file)
        for record_file in set(self.records):
            record_file = os.path.join(self.root_dir, record_file)
            if not os.path.exists(record_file):
                raise RuntimeError("tfrecord file： %s not found" % record_file)
        
        self.get_new_label()
        self.class_num = max(self.labels) + 1
        self.sample_num = len(self.records)
        
        print('class_num: %d, sample_num: %d' % (self.class_num, self.sample_num))
    
    def get_new_label(self):
        label_map = {}
        new_labels = []
        cnt = 0
        for old_label in self.labels:
            if label_map.get(old_label) is None:
                label_map[old_label] = cnt
                cnt += 1
            new_labels.append(label_map[old_label])
        
        self.labels = new_labels

    def __getitem__(self, index):
        offset = self.offsets[index]
        record = self.records[index]
        record_file = os.path.join(self.root_dir, record)
        return self._get_record(record_file, offset), self.labels[index]

class PairIndexTFRDataset(Dataset):
    """Index TFRecord Dataset
    """

    def __init__(self, tfrecord_dir, index_file, transform):
        """Create a ``IndexTFRDataset`` object
        A ``IndexTFRDataset`` object will read sample proto from *.tfrecord files saved
        in ``tfrecord_dir`` by index_file, the sample proto will convert to image and
        fed into Dataloader.

        Args:
            tfrecord_dir: tfrecord saved dir
            index_file: each line format ``tfr_name\t tfr_record_index \t tfr_record_offset \t label
                                           tfr_name\t tfr_record_index \t tfr_record_offset \t label``
            transform: image transform
        """
        self.root_dir = tfrecord_dir
        self.index_file = index_file
        self.transform = transform
        self.records, self.offsets, self.labels = read_pair_index_file(self.index_file)
        records_first, records_second = map(list, zip(*self.records))
        for record_file in set(records_first):
            record_file = os.path.join(self.root_dir, record_file)
            if not os.path.exists(record_file):
                raise RuntimeError("tfrecord file： %s not found" % record_file)
        for record_file in set(records_second):
            record_file = os.path.join(self.root_dir, record_file)
            if not os.path.exists(record_file):
                raise RuntimeError("tfrecord file： %s not found" % record_file)

        self.sample_num = len(self.records)
        first_labels, _ = map(list, zip(*self.labels))
        self.class_num = max(first_labels) + 1
        print('class_num: %d, sample_num:  %d' % (self.class_num, self.sample_num))

    def __len__(self):
        return self.sample_num

    def _parser(self, feature_list):
        for key, feature in feature_list:
            if key == 'image':
                image_raw = feature.bytes_list.value[0]
                image = Image.open(BytesIO(image_raw))
                image = image.convert('RGB')
                image = self.transform(image)
        return image

    def _get_record(self, record_file, offset):
        with open(record_file, 'rb') as ifs:
            ifs.seek(offset)
            byte_len_crc = ifs.read(12)
            proto_len = struct.unpack('Q', byte_len_crc[:8])[0]
            # proto,crc
            pb_data = ifs.read(proto_len)
            if len(pb_data) < proto_len:
                print("read pb_data err,proto_len:%s pb_data len:%s" % (proto_len, len(pb_data)))
                return None
        example = example_pb2.Example()
        example.ParseFromString(pb_data)
        # keep key value in order
        feature = sorted(example.features.feature.items())
        record = self._parser(feature)
        return record

    def __getitem__(self, index):
        first_offset, second_offset = self.offsets[index]
        first_record, second_record = self.records[index]
        first_record_file = os.path.join(self.root_dir, first_record)
        second_record_file = os.path.join(self.root_dir, second_record)
        first_image = self._get_record(first_record_file, first_offset)
        second_image = self._get_record(second_record_file, second_offset)

        return first_image, second_image, self.labels[index][0]
