from md_classic import load_and_run_detector
from PIL import Image, ImageChops
from datetime import datetime
import cv2
import io
from log import log

threshold = 0.4
best_preds = {}


def process_photo(image: Image, camera_id):
    preds, img_output = load_and_run_detector(image=image,
                                              render_confidence_threshold=threshold)

    if camera_id not in best_preds.keys():
        best_preds[camera_id] = {
            'lynx':
                {
                    'value': 0,
                    'image': None
                },
            'brown bear':
                {
                    'value': 0,
                    'image': None
                },
            'moose':
                {
                    'value': 0,
                    'image': None
                },
            'wild boar':
                {
                    'value': 0,
                    'image': None
                },
            'other':
                {
                    'value': 0,
                    'image': None
                }
        }

    for animal in preds.keys():
        if preds[animal] > best_preds[camera_id][animal]['value']:
            best_preds[camera_id][animal]['value'] = preds[animal]
            buf = io.BytesIO()
            img_output.save(buf, format='JPEG')
            best_preds[camera_id][animal]['image'] = buf.getvalue()


def stop_processing(camera_id, area, process_time=None):
    log("from stop_processing")
    result = best_preds.pop(camera_id)
    # log(result)
    predictions = []
    keys = list(result.keys())
    for i in range(len(keys)):
        matches = {'lynx': 0, 'brown bear': 0, 'moose': 0, 'wild boar': 0, 'other': 0}
        animal1 = keys[i]
        log(animal1)
        matches[animal1] = result[animal1]['value']
        log(matches)
        if result[animal1]['value'] > 0:
            for j in range(i + 1, len(keys)):
                animal2 = keys[j]
                log(animal2)
                if result[animal2]['value'] > 0:
                    img_animal_2 = Image.open(io.BytesIO(result[animal2]['image']))
                    img_animal_1 = Image.open(io.BytesIO(result[animal1]['image']))
                    diff = ImageChops.difference(img_animal_2, img_animal_1)
                    if diff.getbbox():
                        matches[animal2] = result[animal2]['value']
            if process_time is None:
                process_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            predictions.append({"camera_id": camera_id,
                                "prediction_time": process_time,
                                "photo": result[animal1]['image'],
                                "moose_count": matches['moose'],
                                "wild_boar_count": matches['wild boar'],
                                "brown_bear_count": matches['brown bear'],
                                "lynx_count": matches['lynx'],
                                "other_count": matches['other'],
                                "area_type": area})
            log(f"PRED\n{predictions}")

    return predictions


def process_video(video, camera_id, area):
    cap = cv2.VideoCapture(video)
    count = 0
    success = True
    while success:
        success, image = cap.read()
        print('read a new frame:', success)
        if count % 30 == 0:
            process_photo(image, camera_id)
        count += 1
    return stop_processing(camera_id, area)
