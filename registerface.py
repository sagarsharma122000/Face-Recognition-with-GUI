import cv2
import numpy as np

face_classifier = cv2.CascadeClassifier('C:/Users/DELL/Desktop/AI/haarcascade_frontalface_default.xml')

def face_extractor(img):
    gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    faces = face_classifier.detectMultiScale(gray,1.3,5)

    if faces is():
        return None
    
    for(x,y,h,w) in faces:
        cropped_face = img[y:y+h,x:x+w]

    return cropped_face

capture = cv2.VideoCapture(0)
count = 0

while True:
    ret,frame = capture.read()
    if face_extractor(frame) is not None:
        count+=1
        face = cv2.resize(face_extractor(frame),(200,200))
        face = cv2.cvtColor(face,cv2.COLOR_BGR2GRAY)

        file_path = 'C:/Users/DELL/Desktop/AI/SAMPLES/'+str(count)+'.jpg'
        cv2.imwrite(file_path,face)

        cv2.putText(face,str(count),(50,50),cv2.FONT_HERSHEY_COMPLEX,1,(0,250,0),2)
        cv2.imshow('Face Cropper',face)
    else:
        print('Face not found')
        pass
    if cv2.waitKey(1)==13 or count==100:
        break

capture.release()
cv2.destroyAllWindows()
print("All samples collected")



        
