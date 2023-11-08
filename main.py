import cv2
from ultralytics import YOLO
import supervision as sv
from line_counter import  *
import numpy as np


LINE_START = sv.Point(20, 240)
LINE_END = sv.Point(620, 240)

video = r'Deployment/files/eggs2.mp4'
video = 0

stop_streaming = False

def egg_detect(filename=''):
    global stop_streaming

    line_counter = LineZone(start=LINE_START, end=LINE_END)
    line_annotator = LineZoneAnnotator(thickness=2, text_thickness=1, text_scale=0.5)

    box_annotator = sv.BoxAnnotator(
        thickness=2,
        text_thickness=1,
        text_scale=0.5
    )

    model = YOLO(r"Deployment/files/best.pt")

    for result in model.track(source=filename, show=False, stream=True, agnostic_nms=True):

        frame = result.orig_img
        detections = sv.Detections.from_yolov8(result)

        if result.boxes.id is not None:
            detections.tracker_id = result.boxes.id.cpu().numpy().astype(int)

        # detections = detections[(detections.class_id != 60) & (detections.class_id != 0)]
        # print("$$$$$$$$$$4",detections)

        detections = detections[(detections.confidence > 0.7)]

        # labels = [
        #   f"{tracker_id} {model.model.names[class_id]} {confidence:0.2f}"
        # E#in detections
        # ]

        def get_label_name(class_id):
            if class_id == 2:
                return "white egg"
            elif class_id == 0:
                return "brown egg"
            elif class_id == 1:
                return "non egg"
            else:
                return "unknown"

        labels = [
            f"{tracker_id} {get_label_name(class_id)} {confidence:0.2f}"
            for _, confidence, class_id, tracker_id
            in detections
        ]

        frame = box_annotator.annotate(
            scene=frame,
            detections=detections,
            labels=labels
        )

        line_counter.trigger(detections=detections)

        line_annotator.annotate(frame=frame, line_counter=line_counter)

        label = "Total Eggs : {}".format(line_counter.out_count)
        white = "White Eggs : {}".format(line_counter.white_count)
        brown = "Brown Eggs : {}".format(line_counter.brown_count)

        cv2.putText(frame, label, (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1, cv2.LINE_AA)
        cv2.putText(frame, white, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1, cv2.LINE_AA)
        cv2.putText(frame, brown, (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1, cv2.LINE_AA)

        if stop_streaming:
            break

        yield  frame

        # cv2.imshow("yolov8", frame)
        #
        # if (cv2.waitKey(30) == 27):
        #     break


if __name__ == "__main__":
    egg_detect(video)

