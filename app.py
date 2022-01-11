import time
from tkinter import *
# from tkinter.ttk import *
from tkinter import scrolledtext
from tkinter import filedialog

from tkinter import Tk, RIGHT, BOTH, RAISED
from tkinter.ttk import Frame, Button, Style,Entry

from subprocess import PIPE, Popen

from os.path import exists as file_exists

prototype_py_file = "live_predict"

class App(Frame):
    def __init__(self):
        super().__init__()
        # master = self.master
        root = self.master
        root.title("live Detection Control Panel")

        # self.window = window
        # window.geometry("300x300+300+300")
        # window.pack(fill=BOTH, expand=True)
        # self.window.title(window_title)

        # self.vdoSource = IntVar()
        # self.optionQuiet = BooleanVar()
        # self.optionQuiet.set(False)
        self.vdo_file_name = StringVar()

        # self.btn_launch.pack(anchor=CENTER, expand=True)
        heading = Label(root, text="Speech Analysis",font=("Arial Bold", 20))
        heading.pack(fill=BOTH, expand=True)
        # self.heading.grid(column=0, row=0)


        master = Frame(root, relief=RAISED)
        master.pack(fill=BOTH, expand=True)

        optionFrame = Frame(master, borderwidth=1)
        # optionFrame.grid(column=0,row=1)
        optionFrame.pack(fill=BOTH,side=LEFT)
        
        # checkQuiet = Checkbutton(optionFrame,text='Suppress face landmark highlight', variable=self.optionQuiet,command=self.optionQuietCheck)
        # radSourceWebcam = Radiobutton(optionFrame,text='Webcam', value=0, variable=self.vdoSource,command=self.sourceClicked)
        # raSourceClip = Radiobutton(optionFrame,text='Video Clip', value=1, variable=self.vdoSource,command=self.sourceClicked)

        # checkQuiet.pack(fill=Y, anchor=W,padx=5, pady=5)
        # radSourceWebcam.pack(fill=Y, anchor=W,padx=5, pady=5)
        # raSourceClip.pack(fill=Y, anchor=W,padx=5, pady=5)
        # self.checkQuiet.grid(column=0, row=2)
        # self.radSourceWebcam.grid(column=0, row=3)
        # self.raSourceClip.grid(column=0, row=4)

        self.btn_open_file = Button(optionFrame, text="Open Video Clip", command=self.openVideoFile,state=NORMAL)        
        # self.btn_open_file.grid(column=1, row=4)
        self.btn_open_file.pack(side=TOP, anchor=W, padx=5, pady=5)

        # lbl_file_name = Label(optionFrame, textvariable=self.vdo_file_name)
        # lbl_file_name.pack(side=LEFT)

        

        logsFrame = Frame(master, borderwidth=1)
        # logsFrame.grid(column=1,row=1)
        logsFrame.pack(fill=Y,side=RIGHT)

        heading = Label(logsFrame, text="Logs",justify='left')
        heading.pack(side=TOP, padx=5, pady=5)

        self.logs = scrolledtext.ScrolledText(logsFrame,width=40,height=10)
        self.logs.insert(INSERT,f"Using {prototype_py_file}\n")
        self.logs.insert(INSERT,"Using Webcam\n")
        self.logs.pack(side=BOTTOM, padx=5, pady=5)
        # self.heading.grid(column=1, row=1)
        # self.logs.grid(column=1,row=2)


        btn_launch = Button(root, text="Launch", width=50,command=self.launch)
        btn_launch.pack(fill=Y,side=BOTTOM, padx=5, pady=5,expand=True)
        # self.btn_launch.grid(column=0, row=8)

        # self.window.mainloop()

    def openVideoFile(self):
        vdo_file_name = filedialog.askopenfilename(filetypes = (("MP4 files","*.mp4"),("WAV files","*.wav")))        
        if file_exists(vdo_file_name):
            print("Ok")
            self.vdo_file_name.set(vdo_file_name)
            self.logs.delete(1.0,END)
            self.logs.insert(INSERT,f"Clip: {vdo_file_name}")
        else:
            print("File Not Found")

    # def optionQuietCheck(self):
    #     print(f"Quiet {self.optionQuiet.get()}")
    #     if self.optionQuiet.get():
    #         self.logs.insert(INSERT,"Disable face landmark highlight\n")
    #     else:
    #         self.logs.insert(INSERT,"Enable face landmark highlight\n")

    def sourceClicked(self):
        print(f"sourceClicked {self.vdoSource.get()}")

        if self.vdoSource.get() == 0: # webcam
            self.logs.delete(1.0,END)
            self.logs.insert(INSERT,"Using Webcam\n")
            self.vdo_file_name.set("")
            self.btn_open_file.configure(state=DISABLED)
        else:
            self.logs.delete(1.0,END)
            self.logs.insert(INSERT,"Choose a clip\n")
            self.btn_open_file.configure(state=NORMAL)

    def update(self):
        pass

    def launch(self):
        cmd = ["python"]
        cmd.append(prototype_py_file)
        # if self.optionQuiet.get():
        #     cmd.append("-q")
        # if self.vdoSource.get() == 1:  # Clip
        #     print(self.vdo_file_name.get())
        #     if self.vdo_file_name.get() == "":
        #         self.logs.insert(INSERT,"Not file selected!\n")
        #         return
        #     else:
        #         cmd.append(self.vdo_file_name.get())               
        if self.vdo_file_name.get() == "":
            self.logs.insert(INSERT,"Not file selected!\n")
            return
        else:
            cmd.append(self.vdo_file_name.get())    
        print(cmd)

        self.logs.delete(1.0,END)
        self.logs.insert(INSERT,f"Running...{' '.join(cmd)}\n")

        command = " ".join(cmd)
        with Popen(command, stdout=PIPE, stderr=None, shell=True) as process:
            output = process.communicate()[0].decode("utf-8")
            self.logs.insert(INSERT,output)

                     
if __name__ == '__main__':
    root = Tk()
    app = App()
    # root.title("DiagFirst Prototype Control Panel")
    root.mainloop()