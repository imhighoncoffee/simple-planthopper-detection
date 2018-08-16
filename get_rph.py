# Thresholding - Binary Thresholding
# Thresholding converts grayscale image into binary

import imutils
import mahotas
# Import Numerical Python package - numpy as np
import numpy as np

# Import Computer Vision package - cv2
import cv2


def nothing(x):
	pass
cv2.namedWindow('img')# crates a window
cv2.createTrackbar('Lower_Threshold', 'img', 119,255,nothing)
cv2.createTrackbar('Upper_Threshold', 'img', 0,255,nothing)
cv2.createTrackbar('H', 'img', 25,255,nothing)
cv2.createTrackbar('S', 'img', 255,255,nothing)
cv2.createTrackbar('V', 'img', 255,255,nothing)
cv2.createTrackbar('h', 'img', 0,255,nothing)
cv2.createTrackbar('s', 'img', 0,255,nothing)
cv2.createTrackbar('v', 'img', 193,255,nothing)
list_of_images = ['MOP.jpg','hop.jpg','POP.jpg','loll.jpg','mom.jpg','poop.jpg','lol.jpg']
for i in list_of_images:
	while(1):
		Lower_Threshold = cv2.getTrackbarPos('Lower_Threshold','img')
		Upper_Threshold= cv2.getTrackbarPos('Upper_Threshold','img')
		H = cv2.getTrackbarPos('H','img')
		S = cv2.getTrackbarPos('S','img')
		V = cv2.getTrackbarPos('V','img')
		h = cv2.getTrackbarPos('h','img')
		s = cv2.getTrackbarPos('s','img')
		v = cv2.getTrackbarPos('v','img')
		# Read the image using imread built-in function
		image = cv2.imread(i)



		# Wait until any key is pressed
		#cv2.waitKey(0)

		# cv2.COLOR_BGR2GRAY: Converts color(RGB) image to gray
		# BGR(bytes are reversed)
		# cv2.cvtColor: Converts image from one color space to another
		HSV = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
		
		lower = np.array([h,s,v])
		higher = np.array([H,S,V])

		mask = cv2.inRange(HSV, lower,higher)

		res = cv2.bitwise_and(image, image, mask= mask)
		gray = cv2.cvtColor(res, cv2.COLOR_BGR2GRAY)
		blurred = cv2.GaussianBlur(gray, (5, 5), 0)



		# cv2.threshold built-in function which performs thresholding
		# cv2.threshold(image, threshold_value, max_value, threshold_type)
		ret,threshold = cv2.threshold(gray, Lower_Threshold, 255, cv2.THRESH_BINARY)
		T=mahotas.thresholding.otsu(blurred)
		b=blurred.copy()
		b[b>T]=255
		b[b<255]=0
		
		kernel = np.ones((5,5),np.uint8)
		opening = cv2.morphologyEx(b,cv2.MORPH_OPEN,kernel)
		closing = cv2.morphologyEx(opening, cv2.MORPH_CLOSE, kernel,iterations = 3)
		
		

		cnts = cv2.findContours(closing.copy(), cv2.RETR_EXTERNAL,
					cv2.CHAIN_APPROX_SIMPLE) 
		cnts = cnts[0] if imutils.is_cv2() else cnts[1]
		area=0
		count=0
		for c in cnts:
			M = cv2.moments(c)
			if M["m00"] > 0:
				cX = int((M["m10"] / M["m00"]+ 1e-7) * 1)
				cY = int((M["m01"] / M["m00"]+ 1e-7) * 1)
				c = c.astype("float")
				c *= 1
				c = c.astype("int")
				area=cv2.contourArea(c)
				if area>100:
					count=count+1
					#cv2.drawContours(image, [c], -1, (0,255,0), 2)
					ellipse = cv2.fitEllipse(c)
					cv2.ellipse(image,ellipse,(0,255,0),2)
					x,y,w,h = cv2.boundingRect(c)
					cv2.putText(image,str(count)+" RPH",(x+5,y+15),cv2.FONT_HERSHEY_PLAIN,1,(255,0,0))

					img=closing[x:y, x+w:y+h]


					




		# Display threshold image using imshow built-in function
		cv2.imshow('Binary Thresholding', b)

		# Display original image using imshow built-in function
		cv2.imshow("Original", image)
		# Wait until any key is pressed
		k = cv2.waitKey(1) & 0xFF
		if k == 27:
			break

# Close all windows
cv2.destroyAllWindows()
