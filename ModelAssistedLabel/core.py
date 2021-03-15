# AUTOGENERATED! DO NOT EDIT! File to edit: 02_augment.ipynb (unless otherwise specified).

__all__ = ['Defaults', 'FileUtilities', 'Generation']

# Cell
class Defaults:
  def __init__(self):
    self.root = "/content/drive/MyDrive/Coding/ModelAssistedLabel/"
    self.resource_folder = "/content/drive/MyDrive/Coding/Roboflow/try it out"
    self.split_ratio = {
              "train": .7,
              "valid": .2,
              "test": .1
            }

# Cell

import glob
from os.path import join

class FileUtilities:
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
    for key, extension in resource_map.items():
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
      for k in resource_map:
        tmp[k] = [x for x in file_collection[k] if x["pair_id"] == resource][0]
      pairs.append(tmp)

    return pairs

  def match_files(walk_dir, recursive=True):
    return FileUtilities.matched(FileUtilities.collect_files(walk_dir, recursive=recursive))

  def mkdir(dir):
    import os
    if not os.path.exists(dir):
      os.mkdir(f"{dir}")


# Cell
from .core import Defaults
from datetime import datetime
import math, random

class Generation:
  """
    Container and organizer of photos for a given repository.
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

  def split(self, split_ratio = None, MAX_SIZE=None):
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

  def write_images(self):
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
            os.system(f"cp {target} {destination}")
    return directories

  def zip_splits_to_destination(self, folder=None):
    assert self.split is not None
    if folder is None:
      folder = self.unified_dirname()
    dirs = self.write_images()
    zipped = self.unify_dirs(folder, dirs)
    os.system(f"mv '{folder}.zip' '{self.out_dir}'")
    return f"{self.out_dir}/{folder}.zip"

  def unify_dirs(self, folder, dirs):
    FileUtilities.mkdir(folder)
    self.write_data_yaml(folder)
    for subdir in self.split:
      os.system(f"mv './{subdir}' '{folder}/'")

    os.system(f'zip -r "{folder}.zip" "{folder}"')
    os.system(f'rm -f -r "{folder}"')
    return f"{folder}.zip"

  def unified_dirname(self, prefix=""):
    now = datetime.now() # current date and time
    timestamp = now.strftime(" %y-%m-%d %H-%M-%S")
    zipname = self.repo.split("/")[-1] + prefix + timestamp
    return zipname

  def write_data_yaml(self, folder="./"):
    f = open(join(folder, "data.yaml"),"w+")
    f.writelines(self.data_yaml)
    f.close()
