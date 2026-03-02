import ultralytics
from ultralytics import YOLO
import cv2

if __name__ == "__main__":
    # Load a model
    model = YOLO('yolo12l.pt')

    # results = model.predict(source='../../../Datasets/data/coco8/images/val/000000000042.jpg', conf=0.25, save=True)
    # results2 = model.predict(source='../../../Datasets/data/coco8/images/val/000000000042.jpg', conf=0.6, save=True)

    results = model.predict(data="VisDrone.yaml", conf=0.25, save=True)

    for r in results:
    # Tespit edilen her kutunun (bounding box) detayları
        for box in r.boxes:
            c = box.cls          # Class Index
            name = model.names[int(c)] # Class Name
            conf = box.conf[0]   # Confidence Score
            coords = box.xyxy[0] # Coordinates (x1, y1, x2, y2) 
        
            print(f"Object: {name} - Confidence: {conf:.2f} - Coordinates: {coords}")

    res_plotted = results[0].plot()  # Plot the first result
    # res_plotted2 = results2[0].plot()  # Plot the second result
    cv2.imshow("YOLOv12l Detection - Conf 0.25", res_plotted)
    # cv2.imshow("YOLOv12l Detection - Conf 0.6", res_plotted2)
    cv2.waitKey(0)
    cv2.destroyAllWindows()