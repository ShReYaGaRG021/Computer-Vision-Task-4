import cv2
model = cv2.CascadeClassifier('haar.xml')

photo = cv2.imread('monica.jpg')
face1 = model.detectMultiScale(photo)
x1 = face1[0][0]
y1 = face1[0][1]
x2 = x1 + face1[0][2]
y2 = y1 + face1[0][3]
a=photo[x1:x2,y1:y2]
#The below line after the face is detected will make a square over the face of the person.
#photo = cv2.rectangle(photo, (x1,y1),(x2,y2),[0,255,0], 5) 
cv2.imshow("photo" , photo)
cv2.waitKey()   
cv2.destroyAllWindows()

pic = cv2.imread('rachel.jpg')
face2 = model.detectMultiScale(pic)
x3 = face2[0][0]
y3 = face2[0][1]
x4 = x3 + face2[0][2]
y4 = y3 + face2[0][3]
b=pic[x3:x4,y3:y4]
#pic = cv2.rectangle(pic, (x3,y3),(x4,y4),[0,255,0], 5)
cv2.imshow("picture", pic)
cv2.waitKey()       
cv2.destroyAllWindows()

p=cv2.resize(b,(a.shape[0],a.shape[1]))
photo[y1:y2,x1:x2]=p
cv2.imshow('new_img',photo)
cv2.waitKey()        
cv2.destroyAllWindows()
