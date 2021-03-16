# AUTOGENERATED! DO NOT EDIT! File to edit: 03_augments.ipynb (unless otherwise specified).

__all__ = ['Defaults', 'FileUtilities', 'Generation', 'Trainer', 'AutoWeights']

# Cell
class Defaults:
  """
  Helps keep a DRY principle across this project. The split ratio was defined based
  on information found on Roboflow. `self.trainer_template` is pulled from the
  YOLOv5 tutorial, as is the code within `prepare_YOLOv5`
  """

  def __init__(self):
    self.root = "/content/drive/MyDrive/Coding/ModelAssistedLabel/"
    self.split_ratio = {
              "train": .7,
              "valid": .2,
              "test": .1
            }
    self.data_yaml = "\n".join(["train: ../train/images",
    "val: ../valid/images",
    "",
    "nc: 10",
    "names: ['1', '2', '3', '4', '5', '6', '7', '8', '9', '0']"])
    self.resource_map = {"images": ".jpg", "labels": ".txt"}

    self.trainer_template = f"""# parameters
    nc: {10}  # number of classes
    depth_multiple: 0.33  # model depth multiple
    width_multiple: 0.50  # layer channel multiple

    # anchors
    anchors:
      - [10,13, 16,30, 33,23]  # P3/8
      - [30,61, 62,45, 59,119]  # P4/16
      - [116,90, 156,198, 373,326]  # P5/32

    # YOLOv5 backbone
    backbone:
      # [from, number, module, args]
      [[-1, 1, Focus, [64, 3]],  # 0-P1/2
      [-1, 1, Conv, [128, 3, 2]],  # 1-P2/4
      [-1, 3, BottleneckCSP, [128]],
      [-1, 1, Conv, [256, 3, 2]],  # 3-P3/8
      [-1, 9, BottleneckCSP, [256]],
      [-1, 1, Conv, [512, 3, 2]],  # 5-P4/16
      [-1, 9, BottleneckCSP, [512]],
      [-1, 1, Conv, [1024, 3, 2]],  # 7-P5/32
      [-1, 1, SPP, [1024, [5, 9, 13]]],
      [-1, 3, BottleneckCSP, [1024, False]],  # 9
      ]

    # YOLOv5 head
    head:
      [[-1, 1, Conv, [512, 1, 1]],
      [-1, 1, nn.Upsample, [None, 2, 'nearest']],
      [[-1, 6], 1, Concat, [1]],  # cat backbone P4
      [-1, 3, BottleneckCSP, [512, False]],  # 13

      [-1, 1, Conv, [256, 1, 1]],
      [-1, 1, nn.Upsample, [None, 2, 'nearest']],
      [[-1, 4], 1, Concat, [1]],  # cat backbone P3
      [-1, 3, BottleneckCSP, [256, False]],  # 17 (P3/8-small)

      [-1, 1, Conv, [256, 3, 2]],
      [[-1, 14], 1, Concat, [1]],  # cat head P4
      [-1, 3, BottleneckCSP, [512, False]],  # 20 (P4/16-medium)

      [-1, 1, Conv, [512, 3, 2]],
      [[-1, 10], 1, Concat, [1]],  # cat head P5
      [-1, 3, BottleneckCSP, [1024,
      False]],  # 23 (P5/32-large)

      [[17, 20, 23], 1, Detect, [nc, anchors]],  # Detect(P3, P4, P5)
      ]"""

  def prepare_YOLOv5():
    """
    * Clone repository if the YOLOv5 directory does not exist.
    * Install requirements.txt
    * Check that GPU is enabled.
    """
    # safety for re-executions
    if not os.path.exists("yolov5"):
      # clone YOLOv5 and reset to a specific git checkpoint that has been verified working
      os.system("git clone https://github.com/ultralytics/yolov5")  # clone repo
      os.system("git reset --hard 68211f72c99915a15855f7b99bf5d93f5631330f") # standardize models

    # enter the yolov5 directory
    os.chdir("yolov5")

    # install dependencies as necessary
    os.system("pip install -qr requirements.txt")  # install dependencies (ignore errors)
    import torch

    from IPython.display import Image, clear_output  # to display images
    # from utils.google_utils import gdrive_download  # to download models/datasets

    clear_output()

    if torch.cuda.is_available():
      print('Setup complete. Using torch %s %s' % (torch.__version__, torch.cuda.get_device_properties(0)))
    else:
      raise Exception("You need to enable your GPU access to this runtime environment")

    # return to parent directory
    os.chdir("..")

  #   calls = {}
  #   calls["nbdev"] = (["pip install nbdev"])
  #   calls["lib/docs"] = (["nbdev_build_lib", "nbdev_build_docs"])
  #   self.calls = calls

  # def call_all(self, arr):
  #   for x in arr:
  #     os.system(x)

  # def nbdev(self):
  #   for x in self.calls:
  #     self.call_all(calls[x])

# Cell

import glob
from os.path import join
import os

class FileUtilities:
  """
  Set of utility functions for managing the requirements of the Ultralytics repo
  """

  def resource_map():
    """
    Explicity define the extensions for images and labels. Check the `Default` class's
    `resource_map` values
    """
    return Defaults().resource_map

  def collect_files(walk_dir, recursive):
    """
    By default, returns all the ".jpg" and ".txt" files in a directory. The filetypes
    are specified by the :resource_map:.

    Args:
      walk_dir: directory from which to pull resources
      recursive: if `True`, resursively searches the folder for the desired resource.

    Returns:
      A dictionary keyed to the :resource_map: with each value being an array of
      the keyed type.
    """
    res = {}
    for key, extension in FileUtilities.resource_map().items():
      resource_generator = glob.iglob(walk_dir + '/**/*' + extension, recursive=recursive)
      res[key] = [{"pair_id": os.path.basename(x)[:-1*len(extension)], "path": x, "basename":os.path.basename(x)} for x in resource_generator]
    return res

  def matched(file_collection):
    """
    Pairs up an image and label based on a shared resource name.

    Arges:
      res: the result of a
    """
    bn = lambda x: set([z["pair_id"] for z in x])
    matched = (bn(file_collection["labels"]).intersection(bn(file_collection["images"])))
    pairs = []
    for resource in matched:
      tmp = {}
      for k in FileUtilities.resource_map():
        tmp[k] = [x for x in file_collection[k] if x["pair_id"] == resource][0]
      pairs.append(tmp)

    return pairs

  def match_files(walk_dir, recursive=True):
    """
    From a bag of resources, find the paired images and labels.

    Args:
      walk_dir: recursively search for images/labels within this folder

    Returns:
      matched pairs of images and text within the `walk_dir`
    """
    return FileUtilities.matched(FileUtilities.collect_files(walk_dir, recursive=recursive))

  def mkdir(dir):
    """
    Don't overwrite an existing file when calling mkdir.
    """
    import os
    if not os.path.exists(dir):
      os.mkdir(f"{dir}")


# Cell
from .core import Defaults
from datetime import datetime
import math, random

class Generation:
  """
    Container and organizer of photos for a given repository. This class "softly"
    organizes the files upon the setting of the `split` attribute via `set_split`.

    The split can then be written to disk by calling `write_split_to_disk`. The
    relevant data will be zipped in `self.out_dir`
  """

  def __init__(self, repo, out_dir, data_yaml):
    """
      Args:
        repo: <string> path to the parent directory of the repository.
    """
    self.repo = repo
    self.split = None
    self.data_yaml = data_yaml
    self.out_dir = out_dir

  def set_split(self, split_ratio = None, MAX_SIZE=None):
    """
    Sets the value of `self.split`

    Args:
      split_ratio: relative fractions of split between test train and validation
      sets.
      MAX_SIZE: The total number of images to be used in the image set
    """
    if split_ratio is None:
      split_ratio = Defaults().split_ratio

    files = FileUtilities.match_files(self.repo)
    random.shuffle(files)
    if MAX_SIZE:
      files = files[:MAX_SIZE]

    train = math.ceil(len(files) * split_ratio["train"])
    valid = train + math.ceil(len(files) * split_ratio["valid"])

    split =  {"train": files[:train],
    "valid": files[train: valid],
    "test": files[valid:]}

    assert sum([len(split[x]) for x in split]) == len(files)
    self.split = split


  def write_split_to_disk(self, descriptor = "", autoname_output=True):
    """
    Takes the given `self.split` and writes the split of the data to disk. Also
    writes a data.yaml file to retain class label information.

    Args:
      descriptor: <str> a unique identifier for the output's filename
      autoname_output: <bool> if True, `descriptor` field is a component of the
      output's filename. Otherwise, it is the entire name.

    Returns:
      A path to the zipped information.
    """
    assert self.split is not None

    if autoname_output:
      out_folder = self.__default_filename__(descriptor)
    else:
      assert len(descriptor) > 0, "need to provide a filename with `descriptor` argument"
      out_folder = descriptor

    dirs = self.__write_images__() #write images
    zipped = self.__zip_dirs__(out_folder, dirs) #zip folders
    os.system(f"mv '{zipped}' '{self.out_dir}'") #move the output
    return f"{self.out_dir}/{zipped}"


  def __zip_dirs__(self, folder, dirs):
    """
    Takes an array of resources and places them all as the children in a specified
    `folder`.

    Args:
      folder: Ultimately will be transformed into `folder.zip`
      dirs: resources to become zipped

    Returns:
      the name of the zip file uniting the resources in `dirs`
    """
    FileUtilities.mkdir(folder)
    self.__write_data_yaml__(folder)
    for subdir in self.split:
      os.system(f"mv './{subdir}' '{folder}/'")

    os.system(f'zip -r "{folder}.zip" "{folder}"')
    os.system(f'rm -f -r "{folder}"')
    return f"{folder}.zip"


  def __write_images__(self):
    """
    If the dataset has already been split, then write the files to disk accordingly.
    All resources are present two levels deep. The top folders are named according
    to "test"/"train"/"valid". The mid-level folders are named "images" or "labels".
    Resources can be found in the corresponding folder.

    Returns:
      A list of directories to the test/train/valid split
    """
    assert self.split is not None
    directories = []
    for dirname, pairs in self.split.items():
      dir = join("./", dirname) #test/valid/train
      FileUtilities.mkdir(dir)
      directories.append(dir)
      for pair in pairs:
        for resource, data in pair.items():
          subdir = join(dir, resource)
          FileUtilities.mkdir(subdir)

          target = data["path"]
          destination = join(subdir, data["basename"])
          print("target/dest", target, "|", destination)
          if not os.path.exists(destination):
            os.system(f"cp '{target}' '{destination}'")
    return directories

  def __default_filename__(self, prefix=""):
    """
    Helper to ease the burden of continually generating unique names or accidentally
    overwriting important data.

    Args:
      prefix: zipfile identifier
    """
    now = datetime.now() # current date and time
    timestamp = now.strftime(" %y-%m-%d %H-%M-%S")
    zipname = self.repo.split("/")[-1] + prefix + timestamp
    return zipname

  def __write_data_yaml__(self, folder, filename="data.yaml"):
    """
    Write `self.data_yaml` to disk.

    Args:
      folder: directory in which to write the data
      filename: optionally rename the yaml data's file
    """
    f = open(os.path.join(folder, filename),"w+")
    f.writelines(self.data_yaml)
    f.close()


# Cell
from .core import Defaults
import os

class Trainer():
  """
  A wrapper for Ultralytic's `test.py`

  Write the backbone of the model to file and then run YOLOv5's train file."""

  def __init__(self, name, yaml_file = "models/custom_yolov5s.yaml"):
    """
    sets the current directory to the project's root as defined in Defaults.

    Args:
      name: identifier for results
      yaml_file: path to write the file
    """
    os.chdir(Defaults().root)
    self.yaml_file = yaml_file
    self.name = name
    self.template = Defaults().trainer_template

  def write_yaml(self):
    """
    Records YOLOv5 architecture
    """
    f = open(f"yolov5/{self.yaml_file}","w+")
    f.writelines(self.template)
    f.close()

  def train(self, epochs):
    """
    wrapper for train.py.

    Args:
      epochs: number of iterations
    """
    self.write_yaml()
    os.chdir("yolov5")
    os.system("pip install -r requirements.txt")
    os.system(f"python train.py --img 416 --batch 16 --epochs {epochs} --data '../data.yaml' --cfg {self.yaml_file} --weights '' --name {self.name}  --cache")
    os.chdir("..")

# Cell
from .core import Trainer, Generation, Defaults
from datetime import datetime

class AutoWeights():
  """
  Given a bag of images (.jpg) and labels (.txt) in YOLOv5 format in a repository,
  initialize the ROOT directory with a train-valid-test split and a file needed
  by the Ultralytics repository. Pairs are identified if `image[:-4] == label[:-4]`

  Then call `generate_weights` to run `train.py`. The resultant file will try to
  be moved to the `out_dir` and if a conflict exists, a new name will be made.
  """
  def __init__(self, resource_dir, name="DEFAULT", out_dir=".", MAX_SIZE=5, custom_data_yaml=None, verbose=True, train_path = "yolov5/runs/train"):
    """
    Args:
      resource_dir: location of the bag of images/labels
      out_dir: where the results of train.py are moved
      MAX_SIZE: parameter for `Generation`
      custom_data_yaml: see `Defaults`'s `data_yaml` for the default value
      verbose: Print summary information
      train_path: path to Ultralytic's default output folder
    """
    self.resource_dir = resource_dir
    self.name = name
    self.resource_paths = ["test/", "train/", "valid/", "data.yaml"]
    for r in self.resource_paths: #make sure none of these paths already exist
      assert os.path.exists(r) is False, f"{r} exists... You may be in the middle of an active project"
    self.out_dir = out_dir
    self.train_path = train_path
    #automatically build the resource paths and prepare for traniing
    self.__prepare_split__(MAX_SIZE=MAX_SIZE, data_yaml=custom_data_yaml, verbose=verbose)
    assert self.g is not None

  def generate_weights(self, epochs, tidy_weights=True):
    """
    Creates a `Trainer` object and trains for a given amount of time.

    Args:
      epochs: number of iterations (according to docs, over 3000 is not uncommon)
      tidy_weights: if True, remove all of the resources in `self.resources`

    Returns:
      path to the output folder of train.py
    """
    t = Trainer(self.name)
    ldir = lambda path: set(os.listdir(path))

    before = ldir(self.train_path)
    t.train(epochs)
    after = ldir(self.train_path)

    assert len(after) == len(before)+1 #only should have made one new file
    diff = list(after - before)[0]

    results_path = os.path.join(self.train_path, diff)

    if tidy_weights:
      results_path = self.__tidy_weights__(results_path = results_path)

    self.__cleanup__()
    self.last_results_path = results_path
    return results_path

  def __prepare_split__(self, MAX_SIZE, data_yaml, verbose):
    """
    Gets the local filesystem ready to run the wrapper for "train.py".

    Args:
      MAX_SIZE:
      data_yaml:
      verbose: print summary information for the split
    """
    if data_yaml is None:
      data_yaml = Defaults().data_yaml
    self.g = Generation(repo=self.resource_dir, out_dir=self.out_dir, data_yaml=data_yaml)
    self.g.set_split(MAX_SIZE=MAX_SIZE)
    self.__split_and_organize_folders__(verbose=verbose)

  def __split_and_organize_folders__(self, verbose):
    """
    Assume zip file contains 4 items:
      * data.yaml
      * train/
      * valid/
      * test/

    Extract these 4 resources to the ROOT directory and remove the original
    part of the file.

    Args:
      verbose: print summary information about the split
    """
    assert self.g is not None

    def tostr(split):
      return [{k: len(v)} for k,v in split.items()]
    def sumg(split):
      return sum([list(x.values())[0] for x in tostr(split)])
    if verbose:
      print("summary: ", tostr(self.g.split))
      print("checksum:", sumg(self.g.split))

    zipped = self.g.write_split_to_disk(self.name) #create a zip file in the ROOT directory
    local = os.path.basename(zipped)
    os.system(f'unzip "{local}"') #grab data
    #move the contents of the zip file into postion within the ROOT directory
    for content in self.resource_paths:
      os.system(f"mv '{local[:-4]}/{content}' .")
    #remove zip file
    os.system(f"rm -f -r '{local}'")
    #removed the folder that was taken out of the zip
    os.system(f"rm -f -r '{local[:-4]}'")

  def __cleanup__(self):
    """
    Removes all resources in `self.resource_paths` from the filesystem.
    """
    for r in self.resource_paths:
      os.system(f"rm -f -r {r}")

  def __tidy_weights__(self, results_path):
    """
    Moves the results to a desired directly while ensuring that no data is overwritten

    Args:
      results_path: path to the folder that has desired information

    Returns:
      Path to the newly-moved results
    """
    while True:
      now = datetime.now()
      out = f"{os.path.basename(results_path)}-{now.strftime('%f')}" #basename + "random" string
      outfolder = os.path.join(self.out_dir, out)
      if not os.path.exists(outfolder): #only stops if a unique name has been found
        break
    os.system(f"mv '{results_path}' '{outfolder}'")
    return outfolder