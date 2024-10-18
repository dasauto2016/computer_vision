import time
import cv2

from ocr3.text_detection import text_detection_and_display

if __name__ == "__main__":
    image_origin = cv2.imread('../photo/test405.png')

    start_time = time.time()
    for i in range(100):
        image = text_detection_and_display(image_origin)
    end_time = time.time()

    print(f"Average execution time per run: {(end_time - start_time) / 100 } seconds")
