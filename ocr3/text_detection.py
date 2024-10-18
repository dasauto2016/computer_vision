import cv2
import pytesseract


def get_grayscale(image):
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


def canny(image):
    return cv2.Canny(image, 100, 200)


def text_detection_and_display(image_origin):
    gray = get_grayscale(image_origin)
    image_handle = canny(gray)

    # # Extract the height and width of the image
    # hImg, wImg = image_handle.shape
    # Define the Tesseract OCR configuration for digit recognition
    config = r'--oem 3 --psm 6 '  # Output digits specifically
    # Use pytesseract to get detailed data about the OCR results, including bounding boxes
    boxs = pytesseract.image_to_data(image_handle, config=config)

    # Iterate over each line of the OCR result data
    for x, b in enumerate(boxs.splitlines()):
        # Skip the first line as it contains headers
        if x != 0:
            # Split the line into individual components
            b = b.split()
            # Check if the line has the expected 12 components (including confidence score)
            if len(b) == 12:
                # Extract the bounding box coordinates (x, y, width, height)
                x, y, w, h = int(b[6]), int(b[7]), int(b[8]), int(b[9])

                # Draw a rectangle around the recognized text on the original image
                cv2.rectangle(image_origin, (x, y), (w + x, h + y), (255, 0, 0), 2)
                # Add the recognized text on top of the bounding box to the original image
                cv2.putText(image_origin, b[11], (x, y + 25), cv2.FONT_HERSHEY_COMPLEX, 0.5, (255, 0, 0), 2)
    return image_origin



if __name__ == "__main__":
    image_origin = cv2.imread('../photo/test405.png')

    image = text_detection_and_display(image_origin)
    # Display the result image with bounding boxes and text overlay
    cv2.imshow("Res", image)
    cv2.waitKey()  # Wait for a key press to close the window
