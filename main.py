import cv2

from detector import Detector
from config import MODEL_PATH, LABELS_PATH
from sort import *



if __name__ == '__main__':

    # Capture video
    cap = cv2.VideoCapture("./data/highway.mp4")

    # Object detector
    detector = Detector(model_path=MODEL_PATH, 
                        labels_path=LABELS_PATH)

    # Tracking
    tracker = Sort(max_age=20, min_hits=3, iou_threshold=0.3)    

    limits_line = [155, 250, 625, 250]
    total_count = []

    while True:
        success, img = cap.read()

        detections = detector.detect(image=img,
                                    detection_classes=['car', 'truck', 'bus', 'motorcycle'])

        results_tracker = tracker.update(detections)

        cv2.line(img, (limits_line[0], limits_line[1]), (limits_line[2], limits_line[3]), (0, 0, 255), 5)
        for result in results_tracker:
            x1, y1, x2, y2, id = result.astype(int)
            w, h = x2 - x1, y2 - y1


            cv2.rectangle(img, 
                          pt1=(x1, y1), 
                          pt2=(x2, y2), 
                          color=(0, 0, 255), 
                          thickness=2, 
                          lineType=9)
            cv2.putText(img, 
                        text=str(id),
                        org=(x1, y1),
                        fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                        fontScale=2,
                        color=(255, 0, 255),
                        thickness=3,
                        lineType=2)

            cx, cy = x1 + w // 2, y1 + h // 2
            cv2.circle(img, 
                       center=(cx, cy), 
                       radius=5, 
                       color=(255, 0, 255), 
                       thickness=cv2.FILLED)

            if limits_line[0] < cx < limits_line[2] and limits_line[1] - 15 < cy < limits_line[1] + 15:
                if total_count.count(id) == 0:
                    total_count.append(id)
                    cv2.line(img, (limits_line[0], limits_line[1]), (limits_line[2], limits_line[3]), (0, 255, 0), 5)

        cv2.putText(img, str(len(total_count)), (255,100), cv2.FONT_HERSHEY_PLAIN,5,(50,50,255),8)

        cv2.imshow('Object tracking', img)
        if cv2.waitKey(1) & 0xff == ord('q'):
            break

