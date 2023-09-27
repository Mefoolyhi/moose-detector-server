import glob
import os
import time
from io import BytesIO
from typing import Union
from PIL import Image, ImageFont, ImageDraw
import humanfriendly
import numpy as np
from utils.augmentations import letterbox
from utils.general import non_max_suppression, xyxy2xywh
from utils.general import scale_boxes as scale_coords
import torch
import math
from shapely.geometry import box
from shapely.ops import unary_union
from torchvision import models
import torchvision.transforms as tt
import sys
from log import log

sys.path.insert(0, './yolov5')
class_names = ['brown bear', 'empty', 'lynx', 'moose', 'wild boar', 'other']

model = models.resnet50(pretrained=True)
model.fc = torch.nn.Linear(model.fc.in_features, 6)
loss = torch.nn.CrossEntropyLoss()
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
checkpoint_path = 'ResNet50_AdamW_1e-05_CosineAnnealingLR_plus_flip.pth'
checkpoint = torch.load(checkpoint_path, map_location=torch.device('cpu'))

model.load_state_dict(checkpoint['model_state_dict'])
model = model.to(device)
model.eval()

val_transforms = tt.Compose([
    tt.Resize((224, 224)),
    tt.ToTensor(),
    tt.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

def truncate_float(x, precision=3):
    assert precision > 0

    if np.isclose(x, 0):
        return 0
    else:
        # Determine the factor, which shifts the decimal point of x
        # just behind the last significant digit
        factor = math.pow(10, precision - 1 - math.floor(math.log10(abs(x))))
        # Shift decimal point by multiplicatipon with factor, flooring, and
        # division by factor
        return math.floor(x * factor) / factor


def truncate_float_array(xs, precision=3):
    """
    Vectorized version of truncate_float(...)

    Args:
    x         (list of float) List of floats to truncate
    precision (int)           The number of significant digits to preserve, should be
                              greater or equal 1
    """

    return [truncate_float(x, precision=precision) for x in xs]


def convert_yolo_to_xywh(yolo_box):
    """
    Converts a YOLO format bounding box to [x_min, y_min, width_of_box, height_of_box].

    Args:
        yolo_box: bounding box of format [x_center, y_center, width_of_box, height_of_box].

    Returns:
        bbox with coordinates represented as [x_min, y_min, width_of_box, height_of_box].
    """
    x_center, y_center, width_of_box, height_of_box = yolo_box
    x_min = x_center - width_of_box / 2.0
    y_min = y_center - height_of_box / 2.0
    return [x_min, y_min, width_of_box, height_of_box]


class PTDetector:
    IMAGE_SIZE = 1280  # image size used in training
    STRIDE = 64

    def __init__(self, model_path: str,
                 use_model_native_classes: bool = False):
        self.device = 'cpu'
        self.model = PTDetector._load_model(model_path, self.device)

        self.printed_image_size_warning = False
        self.use_model_native_classes = use_model_native_classes

    @staticmethod
    def _load_model(model_pt_path, device):
        checkpoint_md = torch.load(model_pt_path, map_location=device)
        model_md = checkpoint_md['model'].float().fuse().eval()  # FP32 model
        return model_md

    def generate_detections_one_image(self, img_original,
                                      detection_threshold):
        result = {}
        detections = []
        max_conf = 0.0
        img_original = np.asarray(img_original)

        # padded resize
        target_size = PTDetector.IMAGE_SIZE

        self.printed_image_size_warning = False

        # ...if the caller has specified an image size

        img = letterbox(img_original, new_shape=target_size,
                        stride=PTDetector.STRIDE, auto=True)[0]  # JIT requires auto=False

        img = img.transpose((2, 0, 1))  # HWC to CHW; PIL Image is RGB already
        img = np.ascontiguousarray(img)
        img = torch.from_numpy(img)
        img = img.to(self.device)
        img = img.float()
        img /= 255

        if len(img.shape) == 3:
            img = torch.unsqueeze(img, 0)

        pred: list = self.model(img)[0]

        pred = non_max_suppression(prediction=pred, conf_thres=detection_threshold)

        # format detections/bounding boxes
        gn = torch.tensor(img_original.shape)[[1, 0, 1, 0]]  # normalization gain whwh

        # This is a loop over detection batches, which will always be length 1 in our case,
        # since we're not doing batch inference.
        for det in pred:

            if len(det):

                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], img_original.shape).round()

                for *xyxy, conf, cls in reversed(det):

                    # normalized center-x, center-y, width and height
                    xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()

                    api_box = convert_yolo_to_xywh(xywh)

                    conf = truncate_float(conf.tolist(), precision=3)

                    cls = int(cls.tolist()) + 1
                    if cls not in (1, 2, 3):
                        raise KeyError(f'{cls} is not a valid class.')

                    detections.append({
                        'category': str(cls),
                        'conf': conf,
                        'bbox': truncate_float_array(api_box, precision=4)
                    })
                    max_conf = max(max_conf, conf)

        result['max_detection_conf'] = max_conf
        result['detections'] = detections

        return result


def load_detector(model_file):
    """Load a TF or PT detector, depending on the extension of model_file."""

    start_time = time.time()
    detector = PTDetector(model_file)
    elapsed = time.time() - start_time
    print('Loaded model in {}'.format(humanfriendly.format_timespan(elapsed)))
    log('LOAD DETECTOR\noaded model in ' + humanfriendly.format_timespan(elapsed))
    return detector


IMAGE_ROTATIONS = {
    3: 180,
    6: 270,
    8: 90
}


def load_image(input_file: Union[str, BytesIO]) -> Image:
    image = Image.open(input_file)
    if image.mode not in ('RGBA', 'RGB', 'L', 'I;16'):
        raise AttributeError(
            f'Image {input_file} uses unsupported mode {image.mode}')
    if image.mode == 'RGBA' or image.mode == 'L':
        image = image.convert(mode='RGB')
    try:
        exif = image._getexif()
        orientation: int = exif.get(274, None)  # 274 is the key for the Orientation field
        if orientation is not None and orientation in IMAGE_ROTATIONS:
            image = image.rotate(IMAGE_ROTATIONS[orientation], expand=True)  # returns a rotated copy
    except Exception:
        pass
    image.load()
    return image


class ImagePathUtils:
    """A collection of utility functions supporting this stand-alone script"""

    # Stick this into filenames before the extension for the rendered result
    DETECTION_FILENAME_INSERT = '_detections'

    image_extensions = ['.jpg', '.jpeg', '.gif', '.png']

    @staticmethod
    def is_image_file(s):
        """
        Check a file's extension against a hard-coded set of image file extensions
        """
        ext = os.path.splitext(s)[1]
        return ext.lower() in ImagePathUtils.image_extensions

    @staticmethod
    def find_image_files(strings):
        """
        Given a list of strings that are potentially image file names, look for strings
        that actually look like image file names (based on extension).
        """
        return [s for s in strings if ImagePathUtils.is_image_file(s)]

    @staticmethod
    def find_images(dir_name, recursive=False):
        """
        Find all files in a directory that look like image file names
        """
        if recursive:
            strings = glob.glob(os.path.join(dir_name, '**', '*.*'), recursive=True)
        else:
            strings = glob.glob(os.path.join(dir_name, '*.*'))

        image_strings = ImagePathUtils.find_image_files(strings)

        return image_strings


detector = load_detector('md_v5b.0.0.pt')


def load_and_run_detector(image,
                          render_confidence_threshold=0.4):
    """Load and run detector on target images, and visualize the results."""

    detection_results = []

    result = detector.generate_detections_one_image(image,
                                                    detection_threshold=0.4)
    detection_results.append(result)

    padding = 50
    print(detection_results)
    log('LOAD & RUN DETECTION\ndetection_results=' + str(detection_results))
    bbox = []
    for detection in detection_results[0]['detections']:
        if int(detection['category']) != 1 or detection['conf'] < render_confidence_threshold:
            continue
        bbox.append(box(detection['bbox'][0], detection['bbox'][1],
                        detection['bbox'][0] + detection['bbox'][2],
                        detection['bbox'][1] + detection['bbox'][3]))

    i = 0
    while i < len(bbox):
        j = -1
        while j < len(bbox) - 1:
            j += 1
            if i == j:
                continue
            iou = bbox[i].intersection(bbox[j]).area / min(bbox[i].area, bbox[j].area)
            if iou > 0.6:
                bbox[i] = unary_union([bbox[i], bbox[j]])
                bbox.pop(j)
        i += 1

    print(bbox)
    log('LOAD & RUN DETECTION\nbbox=' + str(bbox))

    preds = {'lynx': 0, 'brown bear': 0, 'moose': 0, 'wild boar': 0, 'other': 0}
    for b in bbox:
        batch = val_transforms(image).unsqueeze(0)
        prediction = model(batch).squeeze(0).softmax(0)
        class_id = prediction.argmax().item()
        score = prediction[class_id].item()
        category_name = class_names[class_id]
        print(category_name, score)
        log('LOAD & RUN DETECTION\ncategory_name=' + category_name + '\nscore= ' + str(score))
        if category_name in preds.keys():
            preds[category_name] += 1

        draw = ImageDraw.Draw(image)
        im_width, im_height = image.size
        xmin, ymin, xmax, ymax = b.bounds
        (left, right, top, bottom) = (xmin * im_width, xmax * im_width,
                                      ymin * im_height, ymax * im_height)
        draw.line([(left, top), (left, bottom), (right, bottom),
                   (right, top), (left, top)], width=4, fill='RoyalBlue')

        if top > 1.1:
            text_bottom = top
        else:
            text_bottom = bottom + 1.1

        font = ImageFont.truetype("FreeMono.ttf", 14)
        text_width, text_height = font.getsize(category_name)
        text_left = left
        margin = np.ceil(0.05 * text_height)
        draw.rectangle(
            [(text_left, text_bottom - text_height - 2 * margin), (text_left + text_width,
                                                                   text_bottom)],
            fill='RoyalBlue')

        draw.text(
            (text_left + margin, text_bottom - text_height - margin),
            category_name,
            fill='black',
            font=font)

    print(preds)
    log('LOAD & RUN DETECTION\npreds=' + str(preds))
    return preds, image

# ...def load_and_run_detector()
