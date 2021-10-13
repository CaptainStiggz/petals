# import pixellib
# from pixellib.instance import instance_segmentation

# segment_image=instance_segmentation()
# segment_image.load_model("mask_rcnn_coco.h5")
# segment_image.segmentImage("5a/ST05_SE010107.jpg", 
#   extract_segmented_objects=True,
#   save_extracted_objects=True, 
#   show_bboxes=True,
#   output_image_name="output.jpg"
# )

import cv2
import numpy as np
import os
import random
import shutil

def read_file(file):
  img = cv2.imread(file)
  return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

def write_file(img, file):
  return cv2.imwrite(file, img)

def rand_image(dir):
  return read_file(dir+"/"+random.choice(os.listdir(dir)))

# experiment 2
def to_grayscale(img):
  return cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

def find_largest_contour(image):
  """
  This function finds all the contours in an image and return the largest
  contour area.
  :param image: a binary image
  """
  image = image.astype(np.uint8)
  contours,_ = cv2.findContours(
    image,
    cv2.RETR_TREE,
    cv2.CHAIN_APPROX_SIMPLE
  )
  return max(contours, key=cv2.contourArea)

def crop(img, crop):
  x, y, w, h = crop
  print("cropping", w, h)
  return img[y:y+h,x:x+w]

def bbox_contour(contour, pad):
  _x, _y, _w, _h = cv2.boundingRect(contour)
  x, y, w, h = _x, _y, _w, _h
  diff = w - h
  if diff > 0:
    x -= pad
    w += 2 * pad
    y = y - diff // 2 - pad
    h = w
  elif diff < 0:
    y -= pad
    h += 2 * pad
    x = x + diff // 2 - pad
    w = h
  if x < 0 or y < 0:
    print("BBOX TOO CLOSE TO EDGE")
    return None
  if w - h > 2:
    print("BBOX IS NOT SQUARE!")
    return None
  return x,y,w,h

def saturate(img, satval):
  imghsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV).astype("float32")
  h, s, v = cv2.split(imghsv)
  s = s*satval
  s = np.clip(s,0,255)
  imgmer = cv2.merge([h,s,v])
  return cv2.cvtColor(imgmer.astype("uint8"), cv2.COLOR_HSV2RGB)

def contrast(img, contrast):
  clip, k = contrast
  lab= cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
  l, a, b = cv2.split(lab)
  clahe = cv2.createCLAHE(clipLimit=clip, tileGridSize=(k,k))
  cl = clahe.apply(l)
  limg = cv2.merge((cl,a,b))
  return cv2.cvtColor(limg, cv2.COLOR_LAB2RGB)

def process_dir(dir, func, limit = 0):
  out = dir+"_out"
  if os.path.exists(out):
    shutil.rmtree(out)
  os.makedirs(out)

  i = 0
  for file in os.listdir(dir):
    if file.endswith(".jpg"):
      func(dir+"/"+file, out+"/"+file)
      i += 1
      if limit > 0 and i > limit:
        break