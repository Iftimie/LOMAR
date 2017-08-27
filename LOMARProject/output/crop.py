import cv2

for x in range(320):
    image = cv2.imread("track"+str(x)+".png")
    image = image[280:450,50:220]
    cv2.imshow("image",image)
    cv2.waitKey(50)
    cv2.imwrite("track"+str(x)+".png",image)