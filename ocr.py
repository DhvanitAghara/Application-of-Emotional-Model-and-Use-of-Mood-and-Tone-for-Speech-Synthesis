# USAGE
# python ocr.py --image images/example_01.png 
# python ocr.py --image images/example_02.png  --preprocess blur

# import the necessary packages
from PIL import Image
import pytesseract
import argparse
import cv2
import os

pytesseract.pytesseract.tesseract_cmd = 'C:/Program Files (x86)/Tesseract-OCR/Tesseract.exe'

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-p", "--preprocess", type=str, default="thresh",
	help="--name of preprocess")
args = vars(ap.parse_args())






# load the example image and convert it to grayscale
image = cv2.imread("G:/Bisag Internship/tesseract-python/tesseract-python/images/mood.jpg")
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)






# check to see if we should apply thresholding to preprocess the
# image
if args["preprocess"] == "thresh":
	gray = cv2.threshold(gray, 0, 255,
		cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]

# make a check to see if median blurring should be done to remove
# noise
elif args["preprocess"] == "blur":
	gray = cv2.medianBlur(gray, 3)

# write the grayscale image to disk as a temporary file so we can
# apply OCR to it
filename = "{}.png".format(os.getpid())
cv2.imwrite(filename, gray)

# load the image as a PIL/Pillow image, apply OCR, and then delete
# the temporary file
text = pytesseract.image_to_string(Image.open(filename))
os.remove(filename)
print(text)

cv2.imshow("Text Detection",image)
cv2.waitKey(0)

file_name = "G:/Bisag Internship/Generate-Audio-From-Emotions-master/Text-classification/text.txt"
file=open(file_name,'w')
file.write(text)
file.close()


