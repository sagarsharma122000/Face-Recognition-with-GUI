from tkinter import *
top = Tk()
top.title("FACE RECOGNITION")
top.geometry("600x300")

def register_face():
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

    cap = cv2.VideoCapture(0)
    count = 0

    while True:
        ret,frame = cap.read()
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

    cap.release()
    cv2.destroyAllWindows()
    print("samples collected")

        


def verify_face():
    import cv2
    import numpy as np
    from os import listdir
    from os.path import isfile, join

    data_path = 'C:/Users/DELL/Desktop/AI/SAMPLES/'
    onlyfiles = [f for f in listdir(data_path) if isfile(join(data_path,f))]

    Training_Data, Labels = [], []

    for i, files in enumerate(onlyfiles):
        image_path = data_path + onlyfiles[i]
        images = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        Training_Data.append(np.asarray(images, dtype=np.uint8))
        Labels.append(i)

    Labels = np.asarray(Labels, dtype=np.int32)

    model = cv2.face.LBPHFaceRecognizer_create()

    model.train(np.asarray(Training_Data), np.asarray(Labels))

    print("Model Training Complete!!!!!")

    face_classifier = cv2.CascadeClassifier('C:/Users/DELL/Desktop/AI/haarcascade_frontalface_default.xml')

    def face_detector(img, size = 0.5):
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = face_classifier.detectMultiScale(gray,1.3,5)

        if faces is():
            return img,[]

        for(x,y,w,h) in faces:
            cv2.rectangle(img, (x,y),(x+w,y+h),(0,255,255),2)
            roi = img[y:y+h, x:x+w]
            roi = cv2.resize(roi, (200,200))

        return img,roi

    cap = cv2.VideoCapture(0)
    while True:

        ret, frame = cap.read()

        image, face = face_detector(frame)

        try:
            face = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
            result = model.predict(face)

            if result[1] < 500:
                confidence = int(100*(1-(result[1])/300))
                display_string = str(confidence)+'% Confidence it is user'
            


            if confidence > 82:
                cv2.putText(image,display_string,(100,120), cv2.FONT_HERSHEY_COMPLEX,1,(250,120,255),2)
                cv2.putText(image, "Unlocked", (250, 450), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 2)
                cv2.imshow('Face Cropper', image)

            else:
                cv2.putText(image, "Locked", (250, 450), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255), 2)
                cv2.imshow('Face Cropper', image)


        except:
            cv2.putText(image, "Face Not Found", (250, 450), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 0, 0), 2)
            cv2.imshow('Face Cropper', image)
            pass

        if cv2.waitKey(1)==13:
            break


    cap.release()
    cv2.destroyAllWindows()



lb = Label(top,text="Welcome To Facial Recognition",font=("Arial Black",20)).pack(pady=5)

l1 = Button(top,text="Register Face",command=register_face)
l1.pack(pady=25)

l2 = Button(top,text="Verify Face",command=verify_face)
l2.pack(pady=10)

l3 = Button(top,text="Exit",command=exit)
l3.pack(pady=15)

top.mainloop()
