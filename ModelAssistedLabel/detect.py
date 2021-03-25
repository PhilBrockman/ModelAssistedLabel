# AUTOGENERATED! DO NOT EDIT! File to edit: 03_detect.ipynb (unless otherwise specified).

__all__ = ['Detector', 'Viewer']

# Cell
import os, torch
os.chdir("yolov5")
from pathlib import Path
from utils.plots import plot_one_box
from utils.general import check_img_size, non_max_suppression
from utils.datasets import LoadStreams, LoadImages
from utils.general import scale_coords, xyxy2xywh
from utils.torch_utils import select_device
from models.experimental import attempt_load
import cv2, os, base64, json
os.chdir("..")

os.system("pip install pillow")

class Detector:
  """A wrapper for training loading saved YOLOv5 weights

  requirements:
    GPU enabled"""
  def __init__(self, weight_path, conf_threshold = .4, iou_threshold = .45, imgsz = 416, save_dir="save_dir", write_annotated_images_to_disk=False):
    """ Constructor. I pulled the default numeric values above directly from the
    detect.py file. I added the option to save model output to both images and
    to txt files

    Args:
      weight_path: the path to which the saved weights are stored
      conf_threshold: lower bound on the acceptable level of uncertainty for a
                      bounding box
      iou_threshold: IoU helps determine how overlapped two shapes are.
      imgsz: resolution of image to process (assumes square)
      save_dir: where to write annotated images
      write_annotated_images_to_disk: save human-friendly annotated images to disk
    """

    self.weight_path = weight_path
    self.conf_threshold = conf_threshold
    self.iou_threshold = iou_threshold
    self.imgsz = imgsz
    self.device = select_device()
    self.model = attempt_load(self.weight_path, map_location=self.device)  # load FP32 model
    self.names = self.model.module.names if hasattr(self.model, 'module') else self.model.names
    self.imgsz = check_img_size(self.imgsz, s=self.model.stride.max())  # check img_size
    self.half = self.device.type != 'cuda'
    if self.half:
      self.model.half()  # to FP16

    self.write_annotated_images_to_disk = write_annotated_images_to_disk
    self.save_dir = save_dir

  def make_dir(dir):
    """makes a directory provided that the directiory doesn't already exist

    Args:
      dir: Directory to create a path towards
    """
    if not os.path.exists(dir):
      os.makedirs(dir)

  def process_image(self, source):
    """Runs on the model with pre-specified weights an input image. See original
    detect.py for more details

    Args:
      source: A string path to pre-specified weights for the model
      save_unscuffed: create copy of the pre-image

    Reurns:
      A JSON-serializable object encoding bounding box information
    """
    override = None
    if os.path.exists(self.save_dir):
      override = input(f"Save directory '{self.save_dir}' exists. \n 'Enter' to continue, anything else to cancel operation")

    if override is None or len(override) == 0:
      return self.__process_image__(source)

    assert False, "this code shouldn't run"


  #not saving images speed up processing dramatically
  def __process_image__(self, source):
    "helper for process_image"
    results = []
    img = torch.zeros((1, 3, self.imgsz, self.imgsz), device=self.device)  # init img
    _ = self.model(img.half() if self.half else img) if self.device.type != 'cpu' else None  # run once
    dataset = LoadImages(source, img_size=self.imgsz)

    if self.write_annotated_images_to_disk:
      save_dir = Path(self.save_dir)
      Detector.make_dir(save_dir)

    for path, img, im0s, vid_cap in dataset:
      tmp = {}
      tmp["predictions"] = []

      img = torch.from_numpy(img).to(self.device)
      img = img.half() if self.half else img.float()  # uint8 to fp16/32
      img /= 255.0  # 0 - 255 to 0.0 - 1.0

      if img.ndimension() == 3:
        img = img.unsqueeze(0)

      pred = self.model(img, augment=False)[0]
      pred = non_max_suppression(pred, self.conf_threshold, self.iou_threshold, agnostic=False)

      for i, det in enumerate(pred):
        p, s, im0, frame = path, '', im0s, getattr(dataset, 'frame', 0)
        p = Path(p)

        s += '%gx%g ' % img.shape[2:]  # print string
        gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
        if len(det):
          det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()
          for c in det[:, -1].unique():
              n = (det[:, -1] == c).sum()  # detections per class
              s += f'{n} {self.names[int(c)]}s, '  # add to string

          if self.write_annotated_images_to_disk:
            tmp["unscuffed"] = f"{save_dir}/unscuffed-{p.name}"
            cv2.imwrite(tmp["unscuffed"], im0)

          for *xyxy, conf, cls in reversed(det):
            xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
            line = (cls, *xywh, conf) # label format
            tmp["predictions"].append(('%g ' * len(line)).rstrip() % line)

            # save image with bboxes drawn on top
            if self.write_annotated_images_to_disk:
              label = f'{self.names[int(cls)]} {conf:.2f}'
              plot_one_box(xyxy, im0, label=label, color=[0,0,200], line_thickness=5)
              tmp["labeled"] = f"{save_dir}/labeled-{p.name}"
              cv2.imwrite(tmp["labeled"], im0)
          results.append(tmp)

    return results

# Cell
from .detect import Detector
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import pandas as pd
import PIL

class Viewer:
  """ Connects a set of pre-trained weights to an image. Also incorporates
  the human-friendly class labels, as opposed to dealing with the label's index
  """
  def __init__(self, weight_path, class_arr):
    """
    constructor builds the detectors (may be relatively time-intensive) and stores
    the targets

    Args:
      weights_paths: an array of paths to weights
    """
    self.detector = Detector(weight_path=weight_path)
    self.class_arr = class_arr
    self.last_result = []

  def plot_for(self, image, show_labels=True, figsize=(20,10)):
    """ (temporarily) overlay predictions onto the image

    Args:
      image: path to image

    Returns: metadata in the form of an array of dicts
    """
    predictions = self.predict_for(image)["predictions"]

    fig, ax = plt.subplots(figsize=figsize)
    #open/display the image
    im0 = PIL.Image.open(image)
    ax.imshow(im0)

    for prediction in predictions:
      # Create a Rectangle patch
      rect = patches.Rectangle((prediction['x'], prediction['y']), prediction['width'], prediction['height'], linewidth=1, edgecolor='r', facecolor='none')
      if show_labels:
        ax.annotate(prediction["class"], xy=(prediction['x'], prediction['y']-10), color='r', fontsize=20)
      # Add the patch to the Axes
      ax.add_patch(rect)

    plt.show()
    return predictions


  def predict_for(self, image):
    """
    The standard YOLOv5 coordinate format is normed to 1. Need to extract the
    original's image width and height to convert to a standard cartesian plane.

    Args:
      image: path to image

    Returns:
      Convert the predictions converted to a full-scale Cartesian coordinate system.
    """
    assert os.path.exists(image)
    #initialize return
    out = {}
    out["image path"] = image
    out["predictions"] = []

    #process the image in yolov5
    results = self.detector.process_image(image)

    #need height/width to de-norm
    PILim= PIL.Image.open(image)
    width, height = PILim.width, PILim.height

    if len(results) > 0:
      print(">>>", results)
      for prediction in results[0]['predictions']:
        bbox = prediction.split(" ")
        out["predictions"].append({
            "class":       self.class_arr[int(bbox[0])],
            "confidence":               float(bbox[5]),
            "height": int(PILim.height* float(bbox[4])),
            "width":  int(PILim.width * float(bbox[3])),
            "x":      int(PILim.width *(float(bbox[1]) - float(bbox[3])/2)),
            "y":      int(PILim.height*(float(bbox[2]) - float(bbox[4])/2)),
            "yolov5 format": prediction
            })
    return out