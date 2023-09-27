from md_classic import load_and_run_detector
from PIL import Image, ImageChops
from datetime import datetime
import cv2
import io
from log import log

threshold = 0.4
best_preds = {}


def process_photo(image: Image, camera_id, animal):
    preds, img_output = load_and_run_detector(image=image,
                                              render_confidence_threshold=threshold)

    if camera_id not in best_preds.keys():
        best_preds[camera_id] = {
            animal:
                {
                    'value': 0,
                    'image': None
                }
        }
        buf = io.BytesIO()
        img_output.save(buf, format='JPEG')
        best_preds[camera_id][animal]['image'] = buf.getvalue()

    if preds[animal] > best_preds[camera_id][animal]['value']:
        best_preds[camera_id][animal]['value'] = preds[animal]
        buf = io.BytesIO()
        img_output.save(buf, format='JPEG')
        best_preds[camera_id][animal]['image'] = buf.getvalue()


def stop_processing(camera_id, process_time=None):
    result = best_preds.pop(camera_id)
    animal = list(result.keys())[0]
    if process_time is None:
        process_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    prediction = {"camera_id": camera_id, "prediction_time": process_time,
                  "photo": result[animal]['image'],
                  "animal_type": animal,
                  "animal_count": result[animal]['value']}
    print(result)
    log('PREDICTOR\npredictions=' + str(result))
    return prediction


def process_video(video, camera_id, animal):
    cap = cv2.VideoCapture(video)
    count = 0
    success = True
    while success:
        success, image = cap.read()
        print('read a new frame:', success)
        if count % 30 == 0:
            process_photo(image, camera_id, animal)
        count += 1
    return stop_processing(camera_id)
