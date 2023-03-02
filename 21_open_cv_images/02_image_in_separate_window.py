import cv2
img = cv2.imread('../computer_vision/DATA/00-puppy.jpg')
while True:
    cv2.imshow('Puppy',img)
    # If we've waited at least 1ms AND we'we pressed Esc
    if cv2.waitKey(1) & 0xFF == 27:
        break
cv2.destroyAllWindows()
