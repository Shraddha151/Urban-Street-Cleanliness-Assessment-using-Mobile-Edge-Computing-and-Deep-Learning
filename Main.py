import tkinter as tk
from tkinter import Message ,Text
from PIL import Image, ImageTk
import pandas as pd

import tkinter.ttk as ttk
import tkinter.font as font
import tkinter.messagebox as tm
import matplotlib.pyplot as plt
import csv
import numpy as np
from PIL import Image, ImageTk
from tkinter import filedialog
import tkinter.messagebox as tm
import object_detection_runner as PR

bgcolor="#DAF7A6"
bgcolor1="#B7C526"
fgcolor="black"


def Home():
	global window
	def clear():
	    print("Clear1")
	    txt.delete(0, 'end')    
	    txt1.delete(0, 'end')
	    txt2.delete(0, 'end')
	    txt3.delete(0, 'end')
  



	window = tk.Tk()
	window.title("Garbage Detection")

 
	window.geometry('1280x720')
	window.configure(background=bgcolor)
	#window.attributes('-fullscreen', True)

	window.grid_rowconfigure(0, weight=1)
	window.grid_columnconfigure(0, weight=1)
	

	message1 = tk.Label(window, text="Garbage Detection using Deep Learning Algorithm" ,bg=bgcolor  ,fg=fgcolor  ,width=50  ,height=3,font=('times', 30, 'italic bold underline')) 
	message1.place(x=100, y=20)

	lbl = tk.Label(window, text="Select Image Path",width=20  ,height=2  ,fg=fgcolor  ,bg=bgcolor ,font=('times', 15, ' bold ') ) 
	lbl.place(x=100, y=200)
	
	txt = tk.Entry(window,width=20,bg="white" ,fg="black",font=('times', 15, ' bold '))
	txt.place(x=400, y=215)

	

	def browse():
		path=filedialog.askdirectory()
		print(path)
		txt.delete(0, 'end')
		txt.insert('end',path)
		if path !="":
			print(path)
		else:
			tm.showinfo("Input error", "Select DataSet Folder")	

	
	def Predictprocess():
		sym=txt.get()
		if sym != "":
			PR.process(sym)
			tm.showinfo("Output", "Detection Done" )
		else:
			tm.showinfo("Input error", "Select test images folder")

	browse = tk.Button(window, text="Browse", command=browse  ,fg=fgcolor  ,bg=bgcolor1  ,width=20  ,height=1, activebackground = "Red" ,font=('times', 15, ' bold '))
	browse.place(x=650, y=200)

	

	PRbutton = tk.Button(window, text="Predict", command=Predictprocess  ,fg=fgcolor   ,bg=bgcolor1   ,width=14  ,height=2, activebackground = "Red" ,font=('times', 15, ' bold '))
	PRbutton.place(x=900, y=600)

	quitWindow = tk.Button(window, text="Quit", command=window.destroy  ,fg=fgcolor   ,bg=bgcolor1  ,width=15  ,height=2, activebackground = "Red" ,font=('times', 15, ' bold '))
	quitWindow.place(x=1070, y=600)

	window.mainloop()
Home()

