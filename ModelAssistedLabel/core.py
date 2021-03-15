# AUTOGENERATED! DO NOT EDIT! File to edit: 02_augment.ipynb (unless otherwise specified).

__all__ = ['Defaults', 'FileUtilities', 'Generation', 'clean_zip']

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
    self.data_yaml = """train: ../train/images
val: ../valid/images

nc: 10
names: ['1', '2', '3', '4', '5', '6', '7', '8', '9', '0']"""

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


  def process_split(self, descriptor = "", autoname_output=True):
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
      out_folder = self.default_filename(descriptor)
    else:
      assert len(descriptor) > 0, "need to provide a filename with `descriptor` argument"
      out_folder = descriptor

    dirs = self.write_images() #write images
    zipped = self.zip_dirs(out_folder, dirs) #zip folders
    os.system(f"mv '{zipped}' '{self.out_dir}'") #move the output
    return f"{self.out_dir}/{zipped}"


  def zip_dirs(self, folder, dirs):
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
    self.write_data_yaml(folder)
    for subdir in self.split:
      os.system(f"mv './{subdir}' '{folder}/'")

    os.system(f'zip -r "{folder}.zip" "{folder}"')
    os.system(f'rm -f -r "{folder}"')
    return f"{folder}.zip"


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
            os.system(f"cp '{target}' '{destination}'")
    return directories

  def default_filename(self, prefix=""):
    """
    Helper to ease the burden of continually generating unique names or accidentally
    overwriting important data.

    Args:
      prefix:
    """
    now = datetime.now() # current date and time
    timestamp = now.strftime(" %y-%m-%d %H-%M-%S")
    zipname = self.repo.split("/")[-1] + prefix + timestamp
    return zipname

  def write_data_yaml(self, folder="./"):
    f = open(join(folder, "data.yaml"),"w+")
    f.writelines(self.data_yaml)
    f.close()


# Cell
def clean_zip(zip):
  os.system(f'rm -f -r "{zip[:-4]}"')
  os.system(f'rm "{zip}"')