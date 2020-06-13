# Reproduce Datasets

This tutorial describes how to create a dataset suitable for use with deep
learning models in [Google Colab](https://colab.research.google.com), or other
environments where solid state drives (SSD) are unavailable to efficiently read
images into the models.

I recommended running these instructions on a local machine as the intermediate
steps involve manipulating thousands of small files, which is prohibitively slow
in the Google Drive file system. None of the steps described here
require significant computational resources, a laptop should work just fine.

We will write data to a flat LMDB database format, so that thousands of files
are stored into a single `*.lmdb` file. More information about LMDB
can be found at the Wikipedia page [Lightning Memory-Mapped Database](https://en.wikipedia.org/wiki/Lightning_Memory-Mapped_Database). Once the data are in LMDB format, you should
not expect to see additional significant performance benefits (in terms of speed)
while training models by reading data from a SSD versus a traditional HDD.

## 1 Prerequisites

This tutorial assumes that data have already been annotated with pixelwise, i.e.,
semantic segmentation labels. As an example, you may use the data available at
the Team Mussels Google Drive folder `My Drive > Data > ADIG_Labelled_Dataset > train_v120`
to proceed with the tutorial. Right click on the folder `train_v120` and select
download, and the folder will be automatically zipped and downloaded.

__Note:__ Ensure you have ~1GB free disk space. The tutorial assumes the dataset
is downloaded to the folder `/tmp/VOCDevkit/`.

Data can be annotated in any of the following ways

1. LabelMe (manually label with desktop tool written in Python)
2. Figure Eight (setup crowdworker environment)
3. Scale AI (using their API, *this is recommended method for future projects*)
4. GNU Image Manipulation Program (GIMP, recommended for editing PNG-format labels)

Refer to the final report for more detailed usage instructions regarding each of
these. The rest of the document will use the `train_v120` data as an example.
If you have not yet done so, download and unpack the data so that you have a
folder structure like:

```
/tmp/VOCDevkit/train_v120
- JPEGImages
- SegmentationClass
```

Software

To install the software dependencies, I recommend using
[Anaconda](https://www.anaconda.com/products/individual) to create a new Python
environment by following these steps:
```
# Create a new environment using Python version 3.6, with name 'torch14-py36'
conda create -n torch14-py36 python=3.6

# Activate the environment.
# Note: depending on the anaconda version, may need to replace 'source' with 'conda' here
source activate torch14-py36  

# install required packages
conda install pytorch torchvision cpuonly -c pytorch
conda install -c conda-forge jupyterlab
pip install lmdb pyarrow
# At the time of writing, lmdb==0.98, pyarrow==0.16.0
# but this should work with future versions 0.xx.x.
```

## 2 Create Folder Structure
The PyTorch data loaders `class VOCSegmentationLMDB(VisionDataset)` require a
folder structure like:

```
/path/VOCdevkit/VOC2012/
- ImageSets/Segmentation/
- - train.txt
- - trainval.txt (optional)
- - val.txt
- JPEGImages
- SegmentationClass
- class_names.txt
```

__Note:__ This structure is created automatically by running the script `labelme2voc.py`:

`./labelme2voc.py /path/to/json/labels/created/in/labelme /output/path --labels labels.txt`

if using manual labels from LabelMe, but this tutorial skips this step and
assumes we already have labels in PNG format from an arbitrary source.

The `*.txt` files under `ImageSets/Segmentation/`, e.g., `train.txt`, list all of
the file names associated with a particular dataset (i.e. train, val, or trainval).
If you have just downloaded `train_v120`, you will be missing these files.
This is intentional, as I have only uploaded the high resolution originals rather
than thousands of small patches. We will now create the patches from the originals
on the local machine. Before doing so I suggest to copy the folder `train_v120`
(and its contents) and rename as `train_v120_originals`, then rename `train_v120`
as `train_v120_patches` so that you have:
```
- /tmp/VOCDevkit/train_v120_patches
- /tmp/VOCDevkit/train_v120_originals
```

## 3 Extract Patches

Run the Jupyter Notebook `voc-images-and-masks-to-patch.ipynb`, setting the path
variable in Cell 2 to `path = ‘/tmp/VOCDevkit/train_v120_patches`. The assertion
in Cell 3 should pass, indicating that there are 152 original images. Leave all
other settings (e.g. patch width) per their defaults. The notebook will then
extract 81 patches per source image for the PNG masks, and then do the same for
the JPEG images. There will now be 12,464 files in each subfolder.

The last cell before the optional visualization (requires `matplotlib`) writes
file names corresponding to patches to `ImageSets/Segmentation/train.txt`, while
ignoring the "original" high-resolution images.

## 4 Create LMDBs

Finally, we’re ready to create the LMDB file by running the following commands:

- Rename train_v120_patches to VOC2012

- Run the script `folder2lmdb.py` by doing:
`python folder2lmdb.py /tmp/ --split train`
Where the first argument `/tmp/` is a mandatory path to `VOCDevkit/VOC2012` and
`--split` is the dataset type to create (`train`, `val` or `trainval`).

You should see the following output:
```
Loading dataset from /tmp/
Generate LMDB to /tmp/train.lmdb
[0/12312]
[5000/12312]
[10000/12312]
Flushing database ...
```

This should take under one minute to complete. I suggest renaming the `*.lmdb`
files with the dataset version so you remember this in the future. The training
code accepts a version argument and expects lmdb file names to be versioned like
`train_v120.lmdb`.

Validation Set

To create the validation set, you will need to download
`ADIG_Labelled_Dataset > val_v101 > GLNI` and restart the above procedure starting
at Section 2. Remember to change `ImageSets/Segmentation/train.txt` to
`ImageSets/Segmentation/val.txt` in the `voc-images-and-masks-to-patch.ipynb`
notebook, then run: `python folder2lmdb.py /tmp/ --split val`

Trainval Set

Copy the 55 files from each subfolder of `val_v101` (`JPEGImages` and `SegmentationClass`)
to their respective locations in `VOCDevkit/VOC2012`, and re-run the steps
described in Section 3. You should have 16,767 total files in `JPEGImages` and
`SegmentationClass` before running:

`python folder2lmdb.py /tmp/ --split trainval`

If you will train models in Google Colab, you can now upload the
final `*.lmdb` files to Google Drive under
`My Drive > Data > ADIG_Labelled_Dataset > LMDB`, or
whichever path is input to `VOCSegmentationLMDB`.
