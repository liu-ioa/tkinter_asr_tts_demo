# -*- coding: utf-8 -*-
"""
Created on Mon May 30 16:30:34 2022

@author: Lenovo
"""

import warnings
warnings.filterwarnings(action='ignore')
import os
import pygame
import tkinter as tk
from tkinter import messagebox
from tkinter import PhotoImage,Label
import cv2
from PIL import Image, ImageTk
#from ttkbootstrap import *
import ttkbootstrap as ttb
from ttskit import sdk_api
from model.models import speech_model
import torchaudio as ta
import numpy as np
import torch
import threading
from threading import Thread
import wave
import pyaudio
import datetime
import requests
import json
import matplotlib
matplotlib.use('agg')
#import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.pylab import mpl
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk


def video_loop(vd,speech):
    global flag
    flag=1
    a=1
    while flag:
        ref,frame = vd.read()
        if ref:
            cvimage = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pilImage = Image.fromarray(cvimage)
            pilImage = pilImage.resize((1000, 540))
            tkImage =  ImageTk.PhotoImage(image=pilImage)
            MYwindow.Canvas_1.create_image(0,0,anchor='nw',image=tkImage)
            a=a+1
            if a==15:
                pygame.mixer.init()
                pygame.mixer.music.load(speech)  
                pygame.mixer.music.set_volume(0.5) 
                pygame.mixer.music.play()
            MYwindow.frame2.update_idletasks()  #最重要的更新是靠这两句来实现
            MYwindow.frame2.update()
        else:
            flag=0
    vd.release()
    cv2.destroyAllWindows()

def get_fu(path_):
    _wavform, _ = ta.load( path_ )
    _feature = ta.compliance.kaldi.fbank(_wavform, num_mel_bins=40)
    _mean = torch.mean(_feature)
    _std = torch.std(_feature)
    _T_feature =  (_feature - _mean) / _std
    inst_T = _T_feature.unsqueeze(0)
    return inst_T
def speechreco(path_):
    inst_T = get_fu( path_ )
    log_  = model_lo( inst_T )
    _pre_ = log_.transpose(0,1).detach().numpy()[0]
    liuiu = [dd for dd in _pre_.argmax(-1) if dd != 0]
    str_end = ''.join([ num_wor[dd] for dd in liuiu ])
    return str_end

def showWelcome():
    rootwel.overrideredirect(True)
    rootwel.attributes("-alpha", 1)#窗口透明度（1为不透明，0为全透明）
    screen_width = rootwel.winfo_screenwidth() 
    screen_height = rootwel.winfo_screenheight()
    width = 425
    height = 344
    window_size = f'{width}x{height}+{round((screen_width-width)/2)}+{round((screen_height-height)/2)}' #round去掉小数
    rootwel.geometry(window_size)
    if os.path.exists('./Resources/3.png'):
        bm = PhotoImage(file = './Resources/3.png')
        lb_welcomelogo = Label(rootwel, image = bm)
        lb_welcomelogo.bm = bm
        lb_welcomelogo.place(x=-5, y=-5)
        speechreco('temp.wav')
    sdk_api.tts_sdk('欢迎使用配电室数据中心.',speaker='biaobei',output='play')
    global ID
    ID=0

def closeWelcome():
    while(ID):
        root.attributes("-alpha", 0)#窗口透明度
    root.attributes("-alpha", 1)#窗口透明度
    rootwel.destroy()

class  App:
    def __init__(self,root):
        root.title('配电室AI助手')
        root.iconbitmap('./Resources/a.ico')
        root.attributes("-alpha", 0)
        #root.resizable(False,False)
        screen_width = root.winfo_screenwidth() 
        screen_height = root.winfo_screenheight()
        width = 1000
        height = 770
        window_size = f'{width}x{height}+{round((screen_width-width)/2)}+{round((screen_height-height)/2)}' #round去掉小数
        root.geometry(window_size)
        self.root=root
        self.notebook = ttb.Notebook(root)
        self.frame1 = tk.Frame(self.notebook)
        self.frame2 = tk.Frame(self.notebook)
        self.frame3 = tk.Frame(self.notebook)
        self.frame4 = tk.Frame(self.notebook)
        #添加进notebook
        self.notebook.add(self.frame1, text='语音问答')
        self.notebook.add(self.frame2, text='语音合成')
        self.notebook.add(self.frame3, text='人脸检测')
        self.notebook.add(self.frame4, text='实时数据')
        self.notebook.place(relx=0,rely=0.18,relheight=0.82,relwidth=1)
        #主界面
        try:
            url = 'http://wthrcdn.etouch.cn/weather_mini'
            response = requests.get(url, {'city': '北京'})
            result = json.loads(response.content.decode())
            data = result.get('data').get('forecast')
            i=data[0]
        except:
            messagebox.showinfo('通知','联网以使用天气服务')
            root.destroy()
        self.label0=ttb.Label(root,text='北京'+' '+i.get('type')+' '+i.get('high')+' '+i.get('low'),bootstyle="danger")
        self.label0.place(relx=0.38,rely=0.08,relheight=0.03,relwidth=0.2)
        self.label1=ttb.Label(root,text=i.get('fengxiang')+'\t'+i.get('fengli').replace('<![CDATA[', '').replace(']]>', ''),bootstyle="danger")
        self.label1.place(relx=0.42,rely=0.12,relheight=0.03,relwidth=0.2)
        self.time=tk.StringVar()
        self.time.set(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
        self.label=ttb.Label(root,textvariable=self.time,bootstyle="danger")
        self.label.place(relx=0.4,rely=0.04,relheight=0.03,relwidth=0.2)
        self.entry=ttb.Entry(root,width = 20,bootstyle="success")
        self.entry.bind("<Return>",search)
        self.entry.place(relx=0.87,rely=0.04,relheight=0.05,relwidth=0.25,anchor=tk.CENTER)
        label2=ttb.Label(root,text='搜索',bootstyle="info")
        label2.place(relx=0.72,rely=0.04,relheight=0.05,relwidth=0.05,anchor=tk.CENTER)
        MainMenu=tk.Menu(root)
        root.config(menu = MainMenu)
        功能=tk.Menu(MainMenu,tearoff = 0)
        功能.add_command(label="语音问答")
        功能.add_command(label="语音合成")
        功能.add_command(label="人脸检测")
        功能.add_command(label="数据分析")
        MainMenu.add_cascade(label="功能",menu=功能)
        设置=tk.Menu(MainMenu,tearoff = 0)
        主题=tk.Menu(设置,tearoff = 0)
        主题.add_command(label="cosmo",command=self.stylec)
        主题.add_command(label="minty",command=self.stylem)
        主题.add_command(label="pulse",command=self.stylep)
        主题.add_command(label="vapor",command=self.stylev)
        设置.add_cascade(label="主题",menu=主题)
        MainMenu.add_cascade(label="设置",menu=设置)
        MainMenu.add_command(label="帮助")
        MainMenu.add_command(label="关于")
        #frame1
        self.Canvas_1 = ttb.Canvas(self.frame1)
        self.Canvas_1.place(relx=0,rely=0.08,relheight=0.92,relwidth=1)
        self.bu1 = ttb.Button(self.frame1,text = '提问',bootstyle="primary-outline",width = 42)
        self.bu1.configure(command=ask)
        self.bu1.place(relx=0,rely=0,relheight=0.08,relwidth=0.5)
        bu1_2 = ttb.Button(self.frame1, text='结束',bootstyle="primary-outline",width = 42)
        bu1_2.configure(command=answer)
        bu1_2.place(relx=0.5,rely=0,relheight=0.08,relwidth=0.5)
        #frame2
        label2_1=ttb.Label(self.frame2,text='请输入文本',bootstyle="info")
        label2_1.place(relx=0.5,rely=0.53,anchor=tk.CENTER)
        self.entry2=ttb.Entry(self.frame2,width = 45,bootstyle="success")
        self.entry2.place(relx=0.5,rely=0.6,anchor=tk.CENTER)
        bu2_1=ttb.Button(self.frame2,text = '播放',bootstyle="primary-outline",width = 15,command=ttsplay)
        bu2_1.place(relx=0.3,rely=0.75,anchor=tk.CENTER)
        bu2_2=ttb.Button(self.frame2,text = '保存',bootstyle="primary-outline",width = 15,command=ttssave)
        bu2_2.place(relx=0.7,rely=0.75,anchor=tk.CENTER)
        self.Canvas_2 = ttb.Canvas(self.frame2,width=280,heigh=240)
        self.Canvas_2.place(relx=0.51,rely=0.27,anchor=tk.CENTER)
        #frame3
        self.bu3_1=ttb.Button(self.frame3,text = '启动(无法启动则多点几次)',bootstyle="defualt-outline",width = 25,command=facedetection)
        self.bu3_1.place(relx=0.5,rely=0.75,anchor=tk.CENTER)
        label3=ttb.Label(self.frame3,text='设置提醒内容',bootstyle="info")
        label3.place(relx=0.5,rely=0.55,anchor=tk.CENTER)
        self.entry3=ttb.Entry(self.frame3,width = 50,bootstyle="success")
        self.entry3.place(relx=0.5,rely=0.62,anchor=tk.CENTER)
        self.Canvas_3 = ttb.Canvas(self.frame3,width=250,heigh=250)
        self.Canvas_3.place(relx=0.5,rely=0.27,anchor=tk.CENTER)
        #frame4
        #self.Canvas_4 = ttb.Canvas(self.frame4, width=800, height=440, bg='white')
        mpl.rcParams['font.sans-serif'] = ['SimHei']  # 中文显示
        mpl.rcParams['axes.unicode_minus'] = False  # 负号显示
        self.figure = Figure(figsize=(10, 5), dpi=70)
        self.fig1 = self.figure.add_subplot(2, 2, 1)
        self.fig2=self.figure.add_subplot(2, 2, 2)
        self.fig3=self.figure.add_subplot(2, 1, 2)
        
        x = np.arange(-2 * np.pi, 2 * np.pi, 0.1)
        y1 = np.sin(x)
        self.fig3.plot(x, y1, color='red', linewidth=2, label='y=sin(x)', linestyle='--')
        self.fig3.set_title('y=sin(x)')
        self.canvas =FigureCanvasTkAgg(self.figure, master=self.frame4)
        self.canvas.draw()
        #self.canvas.get_tk_widget().place(x=0,y=0)
        self.canvas.get_tk_widget().pack()
        #pack(side=tk.TOP, fill=tk.BOTH, expand=tk.YES).place(relx=0,rely=0,relheight=0.5,relwidth=0.5)
        #把matplotlib绘制图形的导航工具栏显示到tkinter窗口上
        self.toolbar = NavigationToolbar2Tk(self.canvas, self.frame4)
        self.toolbar.update()
      
    def stylec(self):
        style.theme_use('cosmo')
    def stylem(self):
        style.theme_use('minty')
    def stylep(self):
        style.theme_use('pulse')
    def stylev(self):
        style.theme_use('vapor')

        
class Recorder():
  def __init__(self, chunk=1024, channels=1, rate=16000):
    self.CHUNK = chunk
    self.FORMAT = pyaudio.paInt16
    self.CHANNELS = channels
    self.RATE = rate
    self._running = True
    self._frames = []
  def start(self):
    threading._start_new_thread(self.__recording, ())
  def __recording(self):
    self._running = True
    self._frames = []
    p = pyaudio.PyAudio()
    stream = p.open(format=self.FORMAT,
            channels=self.CHANNELS,
            rate=self.RATE,
            input=True,
            frames_per_buffer=self.CHUNK)
    while(self._running):
      data = stream.read(self.CHUNK)
      self._frames.append(data)
  
    stream.stop_stream()
    stream.close()
    p.terminate()
  
  def stop(self):
    self._running = False
  
  def save(self, filename):
     
    p = pyaudio.PyAudio()
    if not filename.endswith(".wav"):
      filename = filename + ".wav"
    wf = wave.open(filename, 'wb')
    wf.setnchannels(self.CHANNELS)
    wf.setsampwidth(p.get_sample_size(self.FORMAT))
    wf.setframerate(self.RATE)
    wf.writeframes(b''.join(self._frames))
    wf.close()
    print("Saved")

def ask():
    global b
    if b==1:
        video_loop(cv2.VideoCapture('./Resources/hello.mp4'),'./Resources/hello.wav')
        b=0
    else:
        video_loop(cv2.VideoCapture('./Resources/wen.mp4'),'./Resources/wen.wav')
    print("Start recording")
    MYwindow.bu1['text']='聆听中'
    rec.start()
def answer():
    print("Stop recording")
    MYwindow.bu1['text']='提问'
    rec.stop()
    rec.save("question.wav")
    path_ = 'question.wav'
    result_ = speechreco(path_)
    print(result_)
    if '正常' in result_:
        sdk_api.tts_sdk('一切设备正常.',speaker='biaobei',output='play')
    elif '名字' in result_:
        video_loop(cv2.VideoCapture('./Resources/name.mp4'),'./Resources/name.wav')
    elif '温度' in result_ or '温' in result_:
        sdk_api.tts_sdk('配电室温度是28摄氏度.',speaker='biaobei',output='play')
    elif '好看' in result_  or '可爱' in result_ or '爱' in result_ :
        video_loop(cv2.VideoCapture('./Resources/haixiu.mp4'),'./Resources/kua.wav')
    elif '漂亮' in result_ or '聪明' in result_ or '聪' in result_:
        video_loop(cv2.VideoCapture('./Resources/thanks.mp4'),'./Resources/thanks.wav')
    elif '笨' in result_ or '傻' in result_ or '啥' in result_ or '猪' in result_:
        video_loop(cv2.VideoCapture('./Resources/nuli.mp4'),'./Resources/nuli.wav')
    elif '你好' in result_ :
        video_loop(cv2.VideoCapture('./Resources/nihao.mp4'),'./Resources/nihao.wav')    
    else:
        video_loop(cv2.VideoCapture('./Resources/notunderstand.mp4'),'./Resources/notunder.wav')
    return 0

def data_updata():  #动态图像现实窗口
    """
    动态matlib图表
    """
    
    MYwindow.fig1.clear()
    y= abs(np.random.normal(1, 2, 12))
    MYwindow.fig1.bar(range(12),abs(y), align='center')
    MYwindow.fig1.set_title("直方图")
    rand_data = np.random.normal(20, 10,10)
    MYwindow.fig2.clear()
    MYwindow.fig2.plot(range(10),rand_data,color='red', linewidth=2,label='wen')
    MYwindow.fig2.set_title("温度曲线图")
    MYwindow.canvas.draw()
    MYwindow.toolbar.update()
    root.after(2000, data_updata)

def time_update():
    MYwindow.time.set(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    root.after(1000,time_update)
def search(event=None):
    string=MYwindow.entry.get()
    print(string)
def ttsplay():
    sdk_api.tts_sdk(MYwindow.entry2.get(),speaker='biaobei',output='play')
def ttssave():
    sdk_api.tts_sdk(MYwindow.entry2.get(),speaker='biaobei',output='./tts/tts.wav')
    messagebox.showinfo('通知','文件已保存至根目录tts文件夹')
def facedetection():
    global q #设置提醒参数
    FaceCascade = cv2.CascadeClassifier('./models/haarcascade_frontalface_default.xml')
    cap = cv2.VideoCapture(0)
    cv2.namedWindow('facedetection',flags=cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO | cv2.WINDOW_GUI_EXPANDED)
    scalar = 2
    while True:
        ret, frame = cap.read()
        if not ret:
            print('video end')
            break
        height, width,_ = frame.shape
        MYwindow.bu3_1['text']='按Esc退出'
        frame_small =  cv2.resize(frame,(int(width/scalar),int(height/scalar)),interpolation=cv2.INTER_CUBIC)
        gray = cv2.cvtColor(frame_small, cv2.COLOR_BGR2GRAY)
        faces = FaceCascade.detectMultiScale(gray,scaleFactor=1.1,minNeighbors=5,)
        if len(faces)>0:
            q=q+1
            if q>65:
                sdk_api.tts_sdk(MYwindow.entry3.get(),speaker='biaobei',output='play')
                q=1
        # 在脸的周围画框框
        for (x, y, w, h) in faces:
        # 从缩放后的ROI，转换为缩放前的ROI
            x *= scalar
            y *= scalar
            w *= scalar
            h *= scalar
            # 绘制画面中人脸区域的矩形
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 4)
            #face_roi = frame[y:y+int(h/2),x:x+w]
            #face_gray = cv2.cvtColor(face_roi, cv2.COLOR_BGR2GRAY)
        cv2.imshow('facedetection', frame)
        if cv2.waitKey(1)==27: 
            MYwindow.bu3_1['text']='启动服务'#& 0xFF == ord('q'):
            break
    # 释放VideoCapture
    #cap.release()
    # 关闭所有的窗口
    cv2.destroyAllWindows()
    
    
model_lo = speech_model()
device_ = torch.device('cpu')
model_lo.load_state_dict(torch.load('models/sp_model.pt' , map_location=device_))
model_lo.eval()
num_wor = np.load('models/dic.dic.npy',allow_pickle=True).item()



if  __name__ == '__main__':
    ID=1 #控制加载界面
    flag=1 #控制视频播放
    b=1 #控制语音助手交互方式
    q=1   #控制人脸检测提醒
    root = ttb.Window(themename="minty")
    style=ttb.Style()
    MYwindow=App(root)
    global rootwel
    rootwel = ttb.Toplevel()
    t1=Thread(target=showWelcome)
    t1.start()
    t2=Thread(target=closeWelcome)
    t2.start()
    
    image=Image.open('./Resources/begin00.png')
    image=image.resize((1000, 540))
    im=ImageTk.PhotoImage(image)
    MYwindow.Canvas_1.create_image(0,0,anchor='nw',image=im)
    
    image1 =Image.open('./Resources/tts.jpeg')
    image1=image1.resize((270, 230))
    image2 = ImageTk.PhotoImage(image1)
    MYwindow.Canvas_2.create_image(0,0,anchor='nw',image=image2)
    
    image3 =Image.open('./Resources/face.jpeg')
    image3=image3.resize((240, 240))
    image3_1 = ImageTk.PhotoImage(image3)
    MYwindow.Canvas_3.create_image(0,0,anchor='nw',image=image3_1)
   
    rec = Recorder()

    time_update()
    data_updata()
    root.protocol('WM_DELETE_WINDOW', root.destroy)
    root.mainloop()














