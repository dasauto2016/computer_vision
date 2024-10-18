import cv2
import pytesseract


def get_grayscale(image):
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


def canny(image):
    return cv2.Canny(image, 100, 200)


def text_detection(image_origin):
    gray = get_grayscale(image_origin)
    image_handle = canny(gray)

    detection_result = []
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
                # Print the recognized text (b[11] is the text content)
                detection_result.append(b[11])

    return detection_result


correct_text = '''
On-campus Recruitment Talk & On-site Interview PricewaterhouseCoopers Hong Kong Date: 18 Oct 2024 (Fri) | Time: 17:00 — 18:00 | Venue: MBGO7, Lingnan University
'''


def effectiveness_rate(detection_result):
    # Split the correct text into words (this assumes correct_text is a string).
    correct_list = correct_text.split(" ")

    # Calculate the number of words in the correct text.
    correct_length = len(correct_list)

    # Calculate the number of "words" (or elements) in the detection result.
    detection_length = len(detection_result)

    # Initialize the total number of alphabets in the correct text to be compared.
    total_alphabet_num = 0

    # Initialize the number of correctly matched alphabets.
    correct_alphabet_num = 0

    # Iterate over the words in both the correct text and detection result,
    # up to the length of the shorter list.
    for x in range(min(correct_length, detection_length)):
        correct_word = correct_list[x]
        detect_word = detection_result[x]

        # Add the length of the current correct word to the total alphabet count.
        total_alphabet_num += len(correct_word)

        # Iterate over the characters in both the current correct word and detected word,
        # up to the length of the shorter word.
        for y in range(min(len(correct_word), len(detect_word))):
            # If the characters at the current position match, increment the correct alphabet count.
            if correct_word[y] == detect_word[y]:
                correct_alphabet_num += 1

    # Calculate and return the effectiveness rate as the ratio of correctly matched alphabets
    # to the total number of alphabets in the correct text.
    return correct_alphabet_num / total_alphabet_num


if __name__ == "__main__":
    image_origin = cv2.imread('../photo/test405.png')
    detection_result = text_detection(image_origin)
    print("the effectiveness_rate is :" + str(effectiveness_rate(detection_result)))
