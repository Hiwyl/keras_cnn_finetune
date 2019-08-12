# -*- coding: utf-8 -*-
'''
@Author :      lance
@Email :   wangyl306@163.com
 '''
from skimage.measure import compare_ssim
#import argparse
import imutils
import cv2

# load the two input images
pathA="errImages/x3-c3/t1_blockId#33255.bmp"
pathB="errImages/x3-c3/t1_blockId#33260.bmp"
imageA = cv2.imread(pathA)
imageB = cv2.imread(pathB)
print(imageA)
print(imageB)
grayA = cv2.imread(pathA,cv2.IMREAD_GRAYSCALE)
grayB = cv2.imread(pathB,cv2.IMREAD_GRAYSCALE)


# compute the Structural Similarity Index (SSIM) between the two
# images, ensuring that the difference image is returned
(score, diff) = compare_ssim(grayA, grayB, full=True)
#score代表两张输入图片的结构相似性索引。
#该值的范围在[-1, 1]，其中值为1时为“完美匹配”
diff = (diff * 255).astype("uint8")
print("SSIM: {}".format(score))

# threshold the difference image, followed by finding contours to
# obtain the regions of the two input images that differ
thresh = cv2.threshold(diff, 0, 255,
	cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL,
	cv2.CHAIN_APPROX_SIMPLE)
cnts = imutils.grab_contours(cnts)

# loop over the contours
for c in cnts:
	# compute the bounding box of the contour and then draw the
	# bounding box on both input images to represent where the two
	# images differ
	(x, y, w, h) = cv2.boundingRect(c)
	cv2.rectangle(imageA, (x, y), (x + w, y + h), (0, 0, 255), 2)
	cv2.rectangle(imageB, (x, y), (x + w, y + h), (0, 0, 255), 2)

# show the output images
cv2.imwrite("errImages/OriginalA.bmp", imageA)
cv2.imwrite("errImages/ModifiedB.bmp", imageB)
cv2.imwrite("errImages/Diff.bmp", diff)
cv2.imwrite("errImages/Thresh.bmp",thresh)
