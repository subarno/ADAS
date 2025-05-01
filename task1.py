import cv2
import numpy as np

img = cv2.imread('Sample1.jpg')
if img is None:
    raise FileNotFoundError("Sample1.jpg not found")

boxes = {
    "RED SUV":   ((70, 390), (220, 480)),
    "WHITE CAR": ((560, 410), (600, 440)),
}

#canny edge and finding roi
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
blur = cv2.GaussianBlur(gray, (5, 5), 0)
edges = cv2.Canny(blur, 50, 150)

h, w = edges.shape
mask = np.zeros_like(edges)
roi_poly = np.array([[
    (0, h), (w, h),
    (w//2 + 50, h//2 + 50),
    (w//2 - 50, h//2 + 50)
]], dtype=np.int32)
cv2.fillPoly(mask, roi_poly, 255)
roi_edges = edges & mask

lines = cv2.HoughLinesP(
    roi_edges, rho=1, theta=np.pi/180,
    threshold=50, minLineLength=50, maxLineGap=100
)

#detecting the cars
cars_only = img.copy()
for tl, br in boxes.values():
    cv2.rectangle(cars_only, tl, br, (0, 0, 255), 2)
cv2.imwrite('cars_only.jpg', cars_only)

#labelling the cars
labelled = cars_only.copy()
font = cv2.FONT_HERSHEY_COMPLEX
for label, (tl, _) in boxes.items():
    x, y = tl
    cv2.putText(labelled, label, (x, y - 10),
                font, 0.6, (255, 0, 0), 2, cv2.LINE_AA)
cv2.imwrite('labelled_cars.jpg', labelled)

#drawing of the lane lines
lane_lines = img.copy()
if lines is not None:
    for x1, y1, x2, y2 in lines.reshape(-1, 4):
        cv2.line(lane_lines, (x1, y1), (x2, y2), (0, 255, 0), 3)
cv2.imwrite('lane_lines.jpg', lane_lines)

#combining all the above outputs on to a single image
combined = lane_lines.copy()
for tl, br in boxes.values():
    cv2.rectangle(combined, tl, br, (0, 0, 255), 2)
for label, (tl, _) in boxes.items():
    x, y = tl
    cv2.putText(combined, label, (x, y - 10),
                font, 0.6, (255, 0, 0), 2, cv2.LINE_AA)
cv2.imwrite('output_assignment1.jpg', combined)

#display all outputs
cv2.imshow("Cars Only", cars_only)
cv2.imshow("Labelled Cars", labelled)
cv2.imshow("Lane Lines", lane_lines)
cv2.imshow("Combined Output", combined)
cv2.waitKey(0)
cv2.destroyAllWindows()