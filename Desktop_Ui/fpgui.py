#! /usr/bin/env python
#  -*- coding: utf-8 -*-
#
# In conjunction with Tcl version 8.6
#    Mar 11, 2019 03:17:24 PM

import sys
from tkinter import filedialog
import PIL
import cv2
from scipy import misc
import numpy as np
import matplotlib.colors as colors
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import threading
from multiprocessing import Process, Manager, Queue
from queue import Empty
from torchvision import transforms
from build_vocab import Vocabulary
from model import EncoderCNN, DecoderRNN
import pickle
import torch
from PIL import ImageTk, Image
import pyttsx3
import time
from gtts import gTTS

try:
    from Tkinter import *
except ImportError:
    from tkinter import *
    import tkinter

try:
    import ttk
    py3 = False
except ImportError:
    import tkinter.ttk as ttk
    py3 = True

import fpgui_support

def vp_start_gui():
    '''Starting point when module is the main routine.'''
    global val, w, root
    root = Tk()
    top = New_Toplevel (root)
    fpgui_support.init(root, top)
    root.mainloop()

w = None
def create_New_Toplevel(root, *args, **kwargs):
    '''Starting point when module is imported by another program.'''
    global w, w_win, rt
    rt = root
    w = Toplevel (root)
    top = New_Toplevel (w)
    fpgui_support.init(w, top, *args, **kwargs)
    return (w, top)

def destroy_New_Toplevel():
    global w
    w.destroy()
    w = None



class New_Toplevel:
    def __init__(self, top=None):
        '''This class configures and populates the toplevel window.
           top is the toplevel containing window.'''
        _bgcolor = '#d9d9d9'  # X11 color: 'gray85'
        _fgcolor = '#000000'  # X11 color: 'black'
        _compcolor = '#d9d9d9' # X11 color: 'gray85'
        _ana1color = '#d9d9d9' # X11 color: 'gray85'
        _ana2color = '#d9d9d9' # X11 color: 'gray85'
        font10 = "-family {DejaVu Sans Mono} -size 9 -weight normal "  \
            "-slant roman -underline 0 -overstrike 0"
        self.style = ttk.Style()
        if sys.platform == "win32":
            self.style.theme_use('winnative')
        self.style.configure('.',background=_bgcolor)
        self.style.configure('.',foreground=_fgcolor)
        self.style.configure('.',font="TkDefaultFont")
        self.style.map('.',background=
            [('selected', _compcolor), ('active',_ana2color)])

        top.geometry("1366x709")
        top.title("New Toplevel")
        top.configure(highlightcolor="black")

        self.bgimage = PIL.Image.open("images.png")
        self.photo_image = PIL.ImageTk.PhotoImage(self.bgimage)
        self.Canvas1 = Canvas(top)
        self.Canvas1.place(relx=0.01, rely=0.01, relheight=0.97, relwidth=0.98)
        self.Canvas1.configure(background="#002a3e")#"#95d6d8")
        self.Canvas1.configure(borderwidth="2")
        self.Canvas1.configure(relief=RAISED)
        self.Canvas1.configure(selectbackground="#c4c4c4")
        self.Canvas1.configure(width=1341)
        self.Canvas1.create_image(570,280, image=self.photo_image, anchor='nw')

        self.Frame2 = Frame(self.Canvas1)
        self.Frame2.place(relx=0.058, rely=0.20, relheight=0.08, relwidth=0.3)
        self.Frame2.configure(borderwidth="2")
        self.Frame2.configure(relief=RAISED)
        self.Frame2.configure(width=365)

        self.Message2 = Message(self.Frame2)
        self.Message2.place(relx=0.23, rely=0.28, relheight=0.3, relwidth=0.5)
        self.Message2.configure(text='''Uploaded Image''')
        self.Message2.configure(width=342)
        self.Message2.configure(foreground="#ff0000")

        self.Frame3 = Frame(self.Canvas1)
        self.Frame3.place(relx=0.65, rely=0.20, relheight=0.08, relwidth=0.3)
        self.Frame3.configure(borderwidth="2")
        self.Frame3.configure(relief=RAISED)
        self.Frame3.configure(width=405)

        self.Message1_1 = Message(self.Frame3)
        self.Message1_1.place(relx=0.23, rely=0.28, relheight=0.3, relwidth=0.5)
        self.Message1_1.configure(text='''HUE SATURATION INTENSITY''')
        self.Message1_1.configure(width=342)
        self.Message1_1.configure(foreground="#ff0000")



        self.Frame1 = Frame(self.Canvas1)
        self.Frame1.place(relx=0.14, rely=0.04, relheight=0.09, relwidth=0.7)
        self.Frame1.configure(borderwidth="2")
        self.Frame1.configure(relief=RAISED)
        self.Frame1.configure(background="#7b1ed8")
        self.Frame1.configure(width=935)

        self.Frame1_1 = Frame(self.Frame1)
        self.Frame1_1.place(relx=0.25, rely=1.78, relheight=0.08, relwidth=0.0)
        self.Frame1_1.configure(borderwidth="2")
        self.Frame1_1.configure(relief=RAISED)
        self.Frame1_1.configure(width=-35)

        self.Message1 = Message(self.Frame1)
        self.Message1.place(relx=0.34, rely=0.15, relheight=0.69, relwidth=0.39)
        self.Message1.configure(foreground="#ff0000", relief = RAISED)
        self.Message1.configure(text='''AUTO IMAGE CAPTIONING''')
        self.Message1.configure(width=362)

        self.Canvas2 = Canvas(self.Canvas1)
        self.Canvas2.place(relx=0.01, rely=0.28, relheight=0.55, relwidth=0.39)
        self.Canvas2.configure(background="#f4ffe6")
        self.Canvas2.configure(borderwidth="2")
        self.Canvas2.configure(relief=RAISED)
        self.Canvas2.configure(selectbackground="#c4c4c4")
        self.Canvas2.configure(width=521)


        self.Canvas3 = Canvas(self.Canvas1)
        self.Canvas3.place(relx=0.6, rely=0.28, relheight=0.55, relwidth=0.39)
        self.Canvas3.configure(background="#f4ffe6")
        self.Canvas3.configure(borderwidth="2")
        self.Canvas3.configure(relief=RAISED)
        self.Canvas3.configure(selectbackground="#c4c4c4")
        self.Canvas3.configure(width=521)

        self.TButton2 = tkinter.Button(self.Canvas1, relief = RAISED)
        self.TButton2.place(relx=0.47, rely=0.94, height=28, width=83)
        self.TButton2.configure(takefocus="")
        self.TButton2.configure(text='''Speak''', command = self.speak())

        self.Entry1 = Entry(self.Canvas1)
        self.Entry1.place(relx=0.28, rely=0.84,height=61, relwidth=0.42)
        self.Entry1.configure(background="white")
        self.Entry1.configure(font=font10)
        self.Entry1.configure(selectbackground="#c4c4c4")

        self.TButton1 = tkinter.Button(self.Canvas1, relief = RAISED)
        self.TButton1.place(relx=0.45, rely=0.74, height=28, width=117)
        self.TButton1.configure(takefocus="")
        self.TButton1.configure(text='''Create Caption''', command = self.start())

        self.TButton3 = tkinter.Button(self.Canvas1, relief = RAISED)
        self.TButton3.place(relx=0.42, rely=0.23, height=48, width=219)
        self.TButton3.configure(takefocus="")
        self.TButton3.configure(text='''Click To Upload An Image''', command = self.open_file())

    def speak(self):

        def act_speak():
            engine = pyttsx3.init() 
            engine.setProperty('voice', 'english')
            engine.setProperty('rate', 135)

            engine.say(self.sentence[7:-5])

            engine.runAndWait()
        return act_speak 

    def start(self):


        def act_start():
            
            return threading.Thread(target=generatecaption(self.path)).start()

        def load_image(image_path, transform=None):
            image = PIL.Image.open(image_path)
            image = image.resize([224, 224], PIL.Image.LANCZOS)

            if transform is not None:
                image = transform(image).unsqueeze(0)

            return image

        def generatecaption(image):
            # Image preprocessing
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406),
                                     (0.229, 0.224, 0.225))])

            # Load vocabulary wrapper
            with open('/root/ImageCaptioning/data/vocab.pkl', 'rb') as f:
                vocab = pickle.load(f)

            # Build models
            encoder = EncoderCNN(256).eval()  # eval mode (batchnorm uses moving mean/variance)
            decoder = DecoderRNN(256, 512, len(vocab), 1)
            encoder = encoder.to(device)
            decoder = decoder.to(device)

            # Load the trained model parameters
            encoder.load_state_dict(torch.load('/root/ImageCaptioning/models/encoder-5-3000.pkl', map_location='cpu'))
            decoder.load_state_dict(torch.load('/root/ImageCaptioning/models/decoder-5-3000.pkl', map_location='cpu'))

            encoder.eval()
            decoder.eval()
            # Prepare an image
            image = load_image(image, transform)
            image_tensor = image.to(device)

            # Generate an caption from the image
            feature = encoder(image_tensor)
            sampled_ids = decoder.sample(feature)
            sampled_ids = sampled_ids[0].cpu().numpy()          # (1, max_seq_length) -> (max_seq_length)

            # Convert word_ids to words
            sampled_caption = []
            for word_id in sampled_ids:
                word = vocab.idx2word[word_id]
                sampled_caption.append(word)
                if word == '<end>':
                    break
            self.sentence = ' '.join(sampled_caption)

            # Print out the image and the generated caption


            self.Entry1.delete(0, END)
            self.Entry1.insert(0,self.sentence[7:-5])

        return act_start


    def open_file(self):

        def return_image():
            self.path=filedialog.askopenfilename(filetypes=[("Image File",'.*')])
            print(self.path)

            img = cv2.imread(self.path)
            newimg = cv2.resize(img,(int(520),int(380)))
            cv2.imwrite("temp/resizeimg.jpg",newimg)
 
            img = misc.imread(self.path)
            array=np.asarray(img)
            arr=(array.astype(float))/255.0
            img_hsv = colors.rgb_to_hsv(arr[...,:3])

            lu1=img_hsv[...,0].flatten()
            plt.subplot(1,3,1)
            plt.hist(lu1*360,bins=360,range=(0.0,360.0),histtype='stepfilled', color='r', label='Hue')
            plt.title("Hue")
            plt.xlabel("Value")
            plt.ylabel("Frequency")

            lu2=img_hsv[...,1].flatten()
            plt.subplot(1,3,2)                  
            plt.hist(lu2,bins=100,range=(0.0,1.0),histtype='stepfilled', color='g', label='Saturation')
            plt.title("Saturation")   
            plt.xlabel("Value")
            plt.yticks()  


            lu3=img_hsv[...,2].flatten()
            plt.subplot(1,3,3)                  
            plt.hist(lu3*255,bins=256,range=(0.0,255.0),histtype='stepfilled', color='b', label='Intesity')
            plt.title("Intensity")   
            plt.xlabel("Value")
            plt.yticks()   

            plt.savefig("temp/plot.png")
            imggraph = cv2.imread("temp/plot.png")
            newimgplot = cv2.resize(imggraph,(550,380))
            cv2.imwrite("temp/resizeimgplot.png",newimgplot)

            self.Canvas2.image = ImageTk.PhotoImage(file = "temp/resizeimg.jpg")
            self.Canvas2.create_image((0,0), image = self.Canvas2.image, anchor = "nw")
            self.Canvas3.image = ImageTk.PhotoImage(file = "temp/resizeimgplot.png")
            self.Canvas3.create_image((0,0), image = self.Canvas3.image, anchor = "nw")

        return return_image

if __name__ == '__main__':
    vp_start_gui()

