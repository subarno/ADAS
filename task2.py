import cv2
import numpy as np

frame = cv2.imread('Sample_02.jpg')
if frame is None:
    raise FileNotFoundError("Make sure 'Sample_02.jpg' is in your working directory")

#grayscale conversion
gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
cv2.imshow('Grayscale', gray)
cv2.imwrite('Grayscale_Sample02.jpg', gray)

#gaussian blur
blur = cv2.GaussianBlur(gray, (5, 5), 0)
cv2.imshow('Gaussian Blur', blur)
cv2.imwrite('Blur_Sample02.jpg', blur)

#canny edge
edges = cv2.Canny(blur, 50, 150)
cv2.imshow('Canny', edges)
cv2.imwrite('Canny_Sample02.jpg', edges)

#segmentation, roi
height, width = edges.shape
mask = np.zeros_like(edges)
poly = np.array([[
    (int(width*0.1), height),
    (int(width*0.9), height),
    (int(width*0.55), int(height*0.6)),
    (int(width*0.45), int(height*0.6))
]], dtype=np.int32)
cv2.fillPoly(mask, poly, 255)
segment = cv2.bitwise_and(edges, mask)
cv2.imshow('Segmented', segment)
cv2.imwrite('Segment_Sample02.jpg', segment)

#hough lines
line_img = np.zeros_like(frame)  # blank color image
lines = cv2.HoughLinesP(segment,
                        rho=1,
                        theta=np.pi/180,
                        threshold=50,
                        minLineLength=100,
                        maxLineGap=50)
if lines is not None:
    for x1, y1, x2, y2 in lines[:, 0]:
        cv2.line(line_img, (x1, y1), (x2, y2), (0, 255, 0), 3)
cv2.imshow('Hough', line_img)
cv2.imwrite('Hough_Sample02.jpg', line_img)

#final output
output = cv2.addWeighted(frame, 1.0, line_img, 1.0, 0)
cv2.imshow('Output Frame', output)
cv2.imwrite('Output_Sample02.jpg', output)

cv2.waitKey(0)
cv2.destroyAllWindows()