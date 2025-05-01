# Lane and Object Edge Detection with OpenCV
### Overview
This project demonstrates two distinct computer vision pipelines using Python and OpenCV:

- Lane Detection: Processes a road image (Sample_02.jpg) through a series of steps—Grayscale conversion, Gaussian Blur, Canny Edge Detection, Region of Interest (ROI) masking, and Hough Line Transform—to detect and overlay lane lines on the original image.
GitHub

- Object Edge Detection: Applies a similar pipeline to a general object image (MyObject.jpg) to extract and visualize object edges.

- Each stage's output is displayed using cv2.imshow() and saved to disk for further analysis.

