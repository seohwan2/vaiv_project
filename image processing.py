import cv2
import numpy as np
file = '/Users/seohwan/Desktop/대학교 과제/Python/fig.png'

img = cv2.imread(file)
imgr= cv2.resize(img, (32, 32))

img_f = imgr.astype(np.float32)
img_pro = ((img_f - img_f.min()) - (img_f - img_f.min()) / (img_f.max() - img_f.min()))
img_pro = img_pro.astype(np.uint8)

img_pro2 = cv2.normalize(imgr, None, 0, 255, cv2.NORM_MINMAX)

cv2.imshow("image_resized", img_pro)

cv2.imshow("image_resized", img_pro2)
cv2.waitKey(0)
cv2.destroyAllWindows()