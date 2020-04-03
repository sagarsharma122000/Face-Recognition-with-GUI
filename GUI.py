from tkinter import *
top = Tk()
top.title("FACE RECOGNITION")
top.geometry("600x300")

def register_face():
def verify_face():
lb = Label(top,text="Welcome To Facial Recognition",font=("Arial Black",20)).pack(pady=5)

l1 = Button(top,text="Register Face",command=register_face)
l1.pack(pady=25)

l2 = Button(top,text="Verify Face",command=verify_face)
l2.pack(pady=10)

l3 = Button(top,text="Exit",command=exit)
l3.pack(pady=15)

top.mainloop()
