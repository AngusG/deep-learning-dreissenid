import io
import os.path as osp
import sys
import tarfile
import collections
from vision import VisionDataset

if sys.version_info[0] == 2:
    import xml.etree.cElementTree as ET
else:
    import xml.etree.ElementTree as ET

from PIL import Image
#from .utils import download_url, check_integrity, verify_str_arg

import six
import string
import argparse

import lmdb
#import pickle
#import msgpack
#import tqdm
import pyarrow as pa

#import torch
import torch.utils.data as data
from torch.utils.data import DataLoader
from torchvision.transforms import transforms
from torchvision.datasets import VOCSegmentation
from torchvision import transforms, datasets


DATASET_YEAR_DICT = {
    '2012': {
        'url': 'http://host.robots.ox.ac.uk/pascal/VOC/voc2012/VOCtrainval_11-May-2012.tar',
        'filename': 'VOCtrainval_11-May-2012.tar',
        'md5': '6cd6e144f989b92b3379bac3b3de84fd',
        'base_dir': osp.join('VOCdevkit', 'VOC2012')
    },
    '2011': {
        'url': 'http://host.robots.ox.ac.uk/pascal/VOC/voc2011/VOCtrainval_25-May-2011.tar',
        'filename': 'VOCtrainval_25-May-2011.tar',
        'md5': '6c3384ef61512963050cb5d687e5bf1e',
        'base_dir': osp.join('TrainVal', 'VOCdevkit', 'VOC2011')
    },
    '2010': {
        'url': 'http://host.robots.ox.ac.uk/pascal/VOC/voc2010/VOCtrainval_03-May-2010.tar',
        'filename': 'VOCtrainval_03-May-2010.tar',
        'md5': 'da459979d0c395079b5c75ee67908abb',
        'base_dir': osp.join('VOCdevkit', 'VOC2010')
    },
    '2009': {
        'url': 'http://host.robots.ox.ac.uk/pascal/VOC/voc2009/VOCtrainval_11-May-2009.tar',
        'filename': 'VOCtrainval_11-May-2009.tar',
        'md5': '59065e4b188729180974ef6572f6a212',
        'base_dir': osp.join('VOCdevkit', 'VOC2009')
    },
    '2008': {
        'url': 'http://host.robots.ox.ac.uk/pascal/VOC/voc2008/VOCtrainval_14-Jul-2008.tar',
        'filename': 'VOCtrainval_11-May-2012.tar',
        'md5': '2629fa636546599198acfcfbfcf1904a',
        'base_dir': osp.join('VOCdevkit', 'VOC2008')
    },
    '2007': {
        'url': 'http://host.robots.ox.ac.uk/pascal/VOC/voc2007/VOCtrainval_06-Nov-2007.tar',
        'filename': 'VOCtrainval_06-Nov-2007.tar',
        'md5': 'c52e279531787c972589f7e41ab4ae64',
        'base_dir': osp.join('VOCdevkit', 'VOC2007')
    }
}


class VOCSegmentationLMDB(VisionDataset):
    """`Pascal VOC <http://host.robots.ox.ac.uk/pascal/VOC/>`_ Segmentation Dataset.

    Args:
        root (string): Root directory of the VOC Dataset.
        year (string, optional): The dataset year, supports years 2007 to 2012.
        image_set (string, optional): Select the image_set to use, ``train``, ``trainval`` or ``val``
        download (bool, optional): If true, downloads the dataset from the internet and
            puts it in root directory. If dataset is already downloaded, it is not
            downloaded again.
        transform (callable, optional): A function/transform that  takes in an PIL image
            and returns a transformed version. E.g, ``transforms.RandomCrop``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
        transforms (callable, optional): A function/transform that takes input sample and its target as entry
            and returns a transformed version.
    """

    def __init__(self,
                 root,
                 download=False,
                 transform=None,
                 target_transform=None,
                 transforms=None):
        super(VOCSegmentationLMDB, self).__init__(root, transforms, transform, target_transform)

        '''
        self.year = year
        self.url = DATASET_YEAR_DICT[year]['url']
        self.filename = DATASET_YEAR_DICT[year]['filename']
        self.md5 = DATASET_YEAR_DICT[year]['md5']
        valid_sets = ["train", "trainval", "val"]
        if year == "2007":
            valid_sets.append("test")
        self.image_set = verify_str_arg(image_set, "image_set", valid_sets)
        base_dir = DATASET_YEAR_DICT[year]['base_dir']
        voc_root = osp.join(self.root, base_dir)
        image_dir = osp.join(voc_root, 'JPEGImages')
        mask_dir = osp.join(voc_root, 'SegmentationClass')

        if download:
            download_extract(self.url, self.root, self.filename, self.md5)

        if not osp.isdir(root):
            raise RuntimeError('Dataset not found or corrupted.' +
                               ' You can use download=True to download it')
        '''
        """
        splits_dir = osp.join(voc_root, 'ImageSets/Segmentation')

        split_f = osp.join(splits_dir, image_set.rstrip('\n') + '.txt')

        with open(osp.join(split_f), "r") as f:
            file_names = [x.strip() for x in f.readlines()]

        self.images = [osp.join(image_dir, x + ".jpg") for x in file_names]
        self.masks = [osp.join(mask_dir, x + ".png") for x in file_names]
        assert (len(self.images) == len(self.masks))
        """
        self.env = lmdb.open(root, subdir=osp.isdir(root),
                             readonly=True, lock=False,
                             readahead=False, meminit=False)

        with self.env.begin(write=False) as txn:
            self.length = loads_pyarrow(txn.get(b'__len__'))
            self.keys = loads_pyarrow(txn.get(b'__keys__'))

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is the image segmentation.
        """

        '''
        img = Image.open(self.images[index]).convert('RGB')
        target = Image.open(self.masks[index])
        '''
        img, target = None, None
        env = self.env
        with env.begin(write=False) as txn:
            byteflow = txn.get(self.keys[index])
        unpacked = loads_pyarrow(byteflow)

        # load image
        imgbuf = unpacked[0]
        byteImgIO = six.BytesIO()
        byteImgIO.write(imgbuf)
        byteImgIO.seek(0)
        img = Image.open(byteImgIO).convert('RGB')

        # load label
        label = unpacked[1]
        byteLabIO = six.BytesIO()
        byteLabIO.write(label)
        byteLabIO.seek(0)
        target = Image.open(byteLabIO)

        if self.transforms is not None:
            img, target = self.transforms(img, target)

        return img, target

    def __len__(self):
        return self.length
        #return len(self.images)

def raw_reader(path):
    with open(path, 'rb') as f:
        bin_data = f.read()
    return bin_data


def dumps_pyarrow(obj):
    """
    Serialize an object.

    Returns:
        Implementation-dependent bytes-like object
    """
    return pa.serialize(obj).to_buffer()


def loads_pyarrow(buf):
    """
    Args:
        buf: the output of `dumps`.
    """
    return pa.deserialize(buf)


def folder2lmdb(dpath, name="train", write_frequency=5000):
    #directory = osp.expanduser(osp.join(dpath, name))
    print("Loading dataset from %s" % dpath)
    #dataset = ImageFolder(directory, loader=raw_reader)

    dataset = VOCSegmentation(
        root=dpath, year='2012', image_set=name, download=False)
    data_loader = DataLoader(dataset, num_workers=4, collate_fn=lambda x: x)

    lmdb_path = osp.join(dpath, "%s.lmdb" % name)
    isdir = osp.isdir(lmdb_path)

    print("Generate LMDB to %s" % lmdb_path)
    db = lmdb.open(lmdb_path, subdir=isdir,
                   map_size=5e9, readonly=False, # 5e8 reserve 500 MB
                   meminit=False, map_async=True)

    txn = db.begin(write=True)

    #https://stackoverflow.com/questions/31077366/pil-cannot-identify-image-file-for-io-bytesio-object
    for idx, data in enumerate(data_loader):
        # print(type(data), data)
        image, label = data[0]

        byteImageIO = io.BytesIO()
        byteLabelIO = io.BytesIO()
        image.save(byteImageIO, "JPEG")
        label.save(byteLabelIO, "PNG")
        byteImageIO.seek(0)
        byteLabelIO.seek(0)
        txn.put(u'{}'.format(idx).encode('ascii'), dumps_pyarrow(
            (byteImageIO.read(), byteLabelIO.read())))
        if idx % write_frequency == 0:
            print("[%d/%d]" % (idx, len(data_loader)))
            txn.commit()
            txn = db.begin(write=True)

    # finish iterating through dataset
    txn.commit()
    keys = [u'{}'.format(k).encode('ascii') for k in range(idx + 1)]
    with db.begin(write=True) as txn:
        txn.put(b'__keys__', dumps_pyarrow(keys))
        txn.put(b'__len__', dumps_pyarrow(len(keys)))

    print("Flushing database ...")
    db.sync()
    db.close()


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    # mandatory arg
    parser.add_argument('path', help='path to VOCDevkit dataset', type=str)
    # optional arg
    parser.add_argument('--split', help='dataset split to do', type=str,
                        default='train', choices=['train', 'val', 'trainval'])

    args = parser.parse_args()

    folder2lmdb(args.path, name=args.split)

    #folder2lmdb("/scratch/ssd/gallowaa/cciw/Lab/", name='train')
