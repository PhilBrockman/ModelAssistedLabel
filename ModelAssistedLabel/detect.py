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

          tmp["predictions"] = []
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

    return tmp

# Cell
from .detect import Detector
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import pandas as pd
import PIL

class Viewer:
  """ After supplying this class with the paths to (1) several pre-generated weights,
  and (2) several new images, you can create a model from each weight and a prediction
  for each image from each model.
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

  def process(self, images, plot_results=True, show_labels=True,figsize=(20,10)):
    """
    Runs an image set through this class's detector.

    Args:
      images: array of images
      plot_results: display the images with superimposed bounding boxes
      show_labels: whether or not to show the human-friendly labels about the bounding boxes
      figsize: passed to plt

    Returns:
      an array of dicts. Dicts relate each set of predictions to an image path
    """
    results = []
    for image in images:
      result = self.detector.process_image(image)
      results.append({"img path": image, "predictions": result})
      #re-calculate predictions with "de-normalized" values
      predictions = self.__yolov5_pred_to_standard__(image, result)["predictions"]
      #show the image with super-imposed bounding boxes
      if plot_results:
        Viewer.__plot_with_bbox__(image,predictions,show_labels=show_labels, figsize=figsize)
    self.last_result.append(results)

  def __plot_with_bbox__(img_path, predictions, show_labels, figsize):
    """display the rectangles on top of the image using pyplot

    Args:
      img_path: path to the image of interest
      predictions: the output from the Detector's process image
      figsize: parameter for fig
    """
    fig, ax = plt.subplots(figsize=figsize)
    #open/display the image
    im0 = PIL.Image.open(img_path)
    ax.imshow(im0)

    for prediction in predictions:
      # Create a Rectangle patch
      rect = patches.Rectangle((prediction['x'], prediction['y']), prediction['width'], prediction['height'], linewidth=1, edgecolor='r', facecolor='none')
      if show_labels:
        ax.annotate(prediction["class"], xy=(prediction['x'], prediction['y']-10), color='r', fontsize=20)
      # Add the patch to the Axes
      ax.add_patch(rect)

    plt.show()

  def __yolov5_pred_to_standard__(self, image, result):
    """
    The standard YOLOv5 coordinate format is normed to 1. Need to extract the
    original's image width and height to convert to a standard cartesian plane.

    Args:
      image: path to image
      result: a dictionary that includes both
          * a key called "filename" that points to the original image
          * a key called "predictions" created when the image is parsed with the
            YOLOv5 model

    Returns:
      Convert the predictions converted to a full-scale Cartesian coordinate system.
    """
    out = {}
    out["image path"] = image
    PILim= PIL.Image.open(image)
    width, height = PILim.width, PILim.height

    out["predictions"] = []
    for prediction in result["predictions"]:
      bbox = prediction.split(" ")
      out["predictions"].append({
          "class": self.class_arr[int(bbox[0])],
          "confidence": float(bbox[5]),
          "height": int(PILim.height*float(bbox[4])),
          "width": int(PILim.width*float(bbox[3])),
          "x": int(PILim.width*(float(bbox[1]) - float(bbox[3])/2)),
          "y": int(PILim.height*(float(bbox[2]) - float(bbox[4])/2))
          })
    return out