import cv2

# 1) Load object image
frame = cv2.imread('object.jpg')
if frame is None:
    raise FileNotFoundError("Cannot load 'object.jpg' – check filename and path")
# 2) Display Input Frame
cv2.namedWindow('Input Frame', cv2.WINDOW_AUTOSIZE)
cv2.imshow('Input Frame', frame)
cv2.imwrite('Input Frame.jpg', frame)

# 3) Grayscale
gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
cv2.namedWindow('Grayscale', cv2.WINDOW_AUTOSIZE)
cv2.imshow('Grayscale', gray)
cv2.imwrite('Grayscale.jpg', gray)

# 4) Gaussian Blur
blur = cv2.GaussianBlur(gray, (5, 5), 0)
cv2.namedWindow('Gaussian Blur', cv2.WINDOW_AUTOSIZE)
cv2.imshow('Gaussian Blur', blur)
cv2.imwrite('Gaussian Blur.jpg', blur)

# 5) Canny Edge Detection
edges = cv2.Canny(blur, 20, 60)
cv2.namedWindow('Canny', cv2.WINDOW_AUTOSIZE)
cv2.imshow('Canny', edges)
cv2.imwrite('Canny.jpg', edges)

# 6) Wait for key press & then clean up
cv2.waitKey(0)
cv2.destroyAllWindows()
