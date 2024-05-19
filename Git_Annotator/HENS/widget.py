import json
import tkinter as tk
from tkinter import ttk, filedialog
import pandas as pd
import os
import numpy as np
import av
# import tkVideoPlayer

from .Feature_Selection import app2
from tkinter.ttk import Style
class Register_Label(ttk.Label):
    def __init__(self, master=None, text=None):
        super().__init__(master=master, text=text)

class Register_Entry(ttk.Entry):
    def __init__(self, master=None, width='30', textvaribale=None):
        super().__init__(master=master, width=width, textvariable=textvaribale)

class Main_Menu(tk.Menu):
    def __init__(self, master=None):
        super().__init__(master=master)

        self.m = tk.Menu(master=self)
        self.file = File_Menu(master=self.m, tearoff=0)
        self.m.add_cascade(label=   '📂file', menu=self.file)
        self.file.add_command(label='📺     Video', command= lambda: self.aaa(master))
        self.file.add_command(label='🕹️ Sensor', command= lambda: self.bbb(master))
        self.file.add_separator()

        self.file.add_command(label='🏭Raw->feature.csv', command=lambda: self.ccc(master))


        # self.file.add_command(label='Register Panel', command=lambda: self.eee(master))
        # self.file.add_separator()


        speedmenu = Speed_Menu(master=self.m, tearoff=0)
        self.m.add_cascade(label='⏩', menu=speedmenu)
        speedmenu.add_radiobutton(label='🐢 X 0.5', command=lambda: self.speed(0.05))
        speedmenu.add_radiobutton(label='🦥 X 0.75', command=lambda: self.speed(0.07))
        speedmenu.add_radiobutton(label='🚶 X 1.0', command=lambda: self.speed(0.0188))
        speedmenu.add_radiobutton(label='🚲 X 1.25', command=lambda: self.speed(0.0126))
        speedmenu.add_radiobutton(label='🚗 X 1.5', command=lambda: self.speed(0.008))
        speedmenu.add_radiobutton(label='✈ X 2', command=lambda: self.speed(0.003))


        batch_menu = Batch_Menu(master=self.m, tearoff=0)
        self.m.add_cascade(label='📍model-fit', menu=batch_menu)
        self.var = tk.IntVar()
        batch_menu.add_radiobutton(label='off', variable=self.var , value=0,command=lambda: self.batch_set('off'))
        batch_menu.add_radiobutton(label='5', variable=self.var, value=5, command=lambda: self.batch_set(5))
        batch_menu.add_radiobutton(label='10',variable=self.var , value=10,command=lambda: self.batch_set(10))
        batch_menu.add_radiobutton(label='20', variable=self.var ,value=20,command=lambda: self.batch_set(20))
        self.var.set(10)


        thresh_menu = Thresh_Menu(master=self.m, tearoff=0)
        self.m.add_cascade(label='🚿threshold', menu=thresh_menu)
        self.var_thresh = tk.DoubleVar()
        thresh_menu.add_radiobutton(label='0.2', variable= self.var_thresh, command=lambda: self.thresh_set(0.2))
        thresh_menu.add_radiobutton(label='0.3', variable= self.var_thresh, value=0.3 ,command=lambda: self.thresh_set(0.3))
        thresh_menu.add_radiobutton(label='0.4',variable= self.var_thresh, command=lambda: self.thresh_set(0.4))
        thresh_menu.add_radiobutton(label='0.5',variable= self.var_thresh, command=lambda: self.thresh_set(0.5))
        thresh_menu.add_radiobutton(label='0.6', variable= self.var_thresh,command=lambda: self.thresh_set(0.6))
        thresh_menu.add_radiobutton(label='0.7',variable= self.var_thresh, command=lambda: self.thresh_set(0.7))
        thresh_menu.add_radiobutton(label='0.8', variable= self.var_thresh,command=lambda: self.thresh_set(0.8))
        thresh_menu.add_radiobutton(label='0.9', variable= self.var_thresh,command=lambda: self.thresh_set(0.9))
        self.var_thresh.set(0.3)

        #24 filter confusing
        filter_menu = Thresh_Menu(master=self.m, tearoff=0)
        self.m.add_cascade(label='Noisy', menu=filter_menu)
        self.var_confu = tk.IntVar()
        filter_menu.add_radiobutton(label='😎Disvisible Confusing Samples', variable=self.var_confu, value=1, command=lambda: self.filter_set(1))
        filter_menu.add_radiobutton(label='👓Visible Confusing Samples', variable=self.var_confu, value=0,command=lambda: self.filter_set(0) )
        self.var_confu.set(0)


        # 6/10 active learning
        self.AL_menu = Batch_Menu(master=self.m, tearoff= 0)
        self.m.add_cascade(label='active learning', menu=self.AL_menu)
        self.var_AL = tk.IntVar()
        self.AL_menu.add_radiobutton(label='🌉Margin', variable = self.var_AL, value = 0 , command= lambda: self.AL_strategy(0))
        self.AL_menu.add_radiobutton(label='🌟Entropy', variable = self.var_AL, value = 1 ,  command= lambda: self.AL_strategy(1))
        self.AL_menu.add_radiobutton(label='🌇Least Confidence', variable = self.var_AL, value = 2 , command = lambda: self.AL_strategy(2))
        self.var_AL.set(0)

        self.analysis_menu = Speed_Menu(master=self.m, tearoff=0)
        self.m.add_cascade(label='🔎 analysis', menu= self.analysis_menu)

        self.model_menu = Speed_Menu(master=self.m, tearoff=0)
        self.m.add_cascade(label='🤖 Estimator', menu=self.model_menu)

        self.var_model = tk.IntVar()
        self.model_menu.add_radiobutton(label='LogisticRegression', variable=self.var_model, value=1, command=lambda: self.Estimating(1))
        self.model_menu.add_radiobutton(label='RandomForest', variable=self.var_model, value=2,
                                command=lambda: self.Estimating(2))
        self.model_menu.add_radiobutton(label='LightGBM', variable=self.var_model, value=3,
                                        command=lambda: self.Estimating(3))
        self.model_menu.add_radiobutton(label='NaiveBayesian', variable=self.var_model, value=4,
                                        command=lambda: self.Estimating(4))
        self.model_menu.add_radiobutton(label='DecisionTree', variable=self.var_model, value=5,
                                        command=lambda: self.Estimating(5))
        self.model_menu.add_radiobutton(label='SVC', variable=self.var_model, value=6,
                                        command=lambda: self.Estimating(6))
        self.var_model.set(1)




        # mode_menu = Mode_Menu(master=self.m, tearoff=0)
        # self.m.add_cascade(label='mode', menu=mode_menu)
        self.label_exist = tk.IntVar()
        self.label_exist.set(True)

        # mode_menu.add_radiobutton(label="With offered data",variable= self.label_exist, value=True,command=lambda: self.mode(True))
        # mode_menu.add_radiobutton(label='No offered label data', variable=self.label_exist, value=False,
        #                           command=lambda: self.mode(False))

        # including data(lab/unlab), video path, video speed(controlled by menu object)
        self.ind_collection = None
        self.X_lab, self.y_lab, self.X_unlab, self.y_unlab, self.start_name, self.end_name, self.video_name, self.identity, self.X_test, self.y_test = None, None, None, None, None, None, None, None, None, None
        self.confusing_X = None
        self.video_path = np.array([])
        self.time_speed = None
        self.batch_size = 10
        self.thresh_hold = 0.3
        self.filter_confusing = 0
        self.V_dict = {}
        self.feature_columns = None
        self.label_columns = None
        self.estimator_ = 'lgr'
        # 4/29
        self.Series = tk.StringVar()
        self.Series2 = tk.StringVar()



    # def mode(self, state):
    #     self.label_exist.set(state)
    #     print(state)
    def AL_strategy(self, num):
        self.var_AL.set(num)
        print(num)

    def Estimating(self, num):
        if num == 1:
            self.estimator_ = 'lgr'
        elif num == 2:
            self.estimator_ = 'rf'
        elif num == 3:
            self.estimator_ = 'lgbm'
        elif num == 4:
            self.estimator_ = 'nb'

        elif num == 5:
            self.estimator_ = 'dt'
        elif num == 6:
            self.estimator_ = 'svc'






    def V_retrive(self, dict):
        print("第一个:",dict)
        t = {}
        for i,value in dict.items():
            time = value['HH'].get()+":"+value['MIN'].get()+":"+value['SEC'].get()+"."+value['MSEC'].get()
            t[i] = time
        self.V_dict = t

    def aaa(self, master):
        print("芝麻开门")
        self.video_path = filedialog.askopenfilenames()
        if self.video_path:
            window = tk.Toplevel(master)
            window.geometry('500x300')
            window.title("Video Start time setting")
            window.resizable(height=True,width=False)


            f1 = ttk.LabelFrame(window, width=300, height=200 )
            f1.pack(side=tk.TOP )
            f1.pack_propagate(True)
            Time_dict = {}
            tk.Label(f1, text='format( Hour:Min:Sec.micro-Sec )\n Example:  10:58:02.33').grid(row=0, columnspan=9)

            for count, i in enumerate(self.video_path):
                text = str(os.path.basename(i))
                # a 仅仅提供数据格式
                a = {"HH": tk.StringVar(), "MIN": tk.StringVar(), "SEC": tk.StringVar(), "MSEC": tk.StringVar()}
                tk.Label(master=f1, text=text+" : ").grid(row=count+1 , column= 0, pady=10)
                tk.Entry(master=f1,width=3,textvariable=a['HH']).grid(row=count+1, column=1)
                tk.Label(master=f1, text=" : ").grid(row=count+1, column=2)
                tk.Entry(master=f1,width=3,textvariable=a['MIN']).grid(row=count+1, column=3)
                tk.Label(master=f1, text=" : ").grid(row=count+1, column=4)
                tk.Entry(master=f1,width=3,textvariable=a['SEC']).grid(row=count+1, column=5)
                tk.Label(master=f1, text=" . ").grid(row=count+1, column=6)
                tk.Entry(master=f1,width=3,textvariable=a['MSEC']).grid(row=count+1, column=7)
                Time_dict[text] = a
            tk.Button(f1, text='⚪', command = lambda: self.V_retrive(Time_dict)).grid(column=9)
    def bbb(self, master):
        print(master, "成功打开!!!")
        from tkinter.ttk import Separator
        window = tk.Toplevel(master)
        window.geometry('1230x630')
        window.title("feature")
        f1 = tk.Frame(window, width=300, height=600)#, bg='#E0E0EE')
        f2 = tk.Frame(window, width=1000, height=600)

        f1.grid(row=0, column=0)
        f2.grid(row=0, column=2)
        f1.grid_propagate(flag=False)
        f2.grid_propagate(flag=False)
        window.grid_propagate(flag=False)

        split = Separator(window,orient='vertical').grid(row=0, column=1, ipady=300, padx=10, pady=10)

        # f1 三个按钮以及文本区域, Initial , train , test
        bt1 = tk.Button(f1, text='initial', command=lambda: in_data(text1, f3))
        bt2 = tk.Button(f1, text='train', command=lambda: train_data(text2, f4))
        bt3 = tk.Button(f1, text='test', command=lambda: test_data(text3, f5))
        bt1.grid(row=0, column=1, ipadx=10, ipady=0, padx=0, pady=0, sticky='N')
        bt2.grid(row=1, column=1, ipadx=10, ipady=0, padx=0, pady=0, sticky='N')
        bt3.grid(row=2, column=1, ipadx=10, ipady=0, padx=0, pady=0, sticky='N')

        text1 = tk.Text(f1, width=34, height=15)
        text2 = tk.Text(f1, width=34, height=15)
        text3 = tk.Text(f1, width=34, height=15)
        text1.grid(row=0, column=0)
        text2.grid(row=1, column=0)
        text3.grid(row=2, column=0)

        # f2为总区域，  f3, f4，f5附属区域 分别为格式化区域
        f3 = tk.Frame(f2, width=1400, height=198)#,   bg='#EEEEE8')#, bg='#CDCDC0')
        f4 = tk.Frame(f2, width=1400, height=198)#,   bg='#EEEEE8')#, bg='#CDCDC0')
        f5 = tk.Frame(f2, width=1400, height=198)#,   bg='#EEEEE8')#, bg='#CDCDC0')
        f3.grid_propagate(False)
        f4.grid_propagate(False)
        f5.grid_propagate(False)
        f3.grid(row=1, column=0, sticky='n')
        f4.grid(row=3, column=0, sticky='n')
        f5.grid(row=5, column=0, sticky='n')
        split2 = Separator(f2, orient='horizontal').grid(row=0, column=0, ipadx=400, padx=10, pady=1, sticky='w')

        split2 = Separator(f2, orient='horizontal').grid(row=2, column=0, ipadx=400,padx=10 , pady=1 , sticky='w' )
        split3 = Separator(f2, orient='horizontal').grid(row=4, column=0, ipadx=400,padx=10 , pady=1, sticky='w')



        # 首先f3中的初始化数据集
        label_x = tk.Label(f3, text='X')
        label_x.grid(row=0, column=0, ipadx=3, padx=206, pady=4)
        label_y = tk.Label(f3, text='Y')
        label_y.grid(row=0, column=1, ipadx=3)

        # 然后,f4区域内的初始化数据按钮
        f4_label_x = tk.Label(f4, text='X')
        f4_label_y = tk.Label(f4, text='Y')
        ## + properties
        f4_start = tk.Label(f4, text='Time_Start')
        f4_end = tk.Label(f4, text='Time_End')
        f4_video = tk.Label(f4, text='Video')
        f4_color = tk.Label(f4, text='Identity_COLOR')

        f4_label_x.grid(row=0, column=0, sticky='W', padx=61)
        f4_label_y.grid(row=0, column=1, sticky='W', padx=61)
        f4_start.grid(row=0, column=2, sticky='N', padx=30)
        f4_end.grid(row=0, column=3, sticky='N', padx=0)
        f4_video.grid(row=0, column=4, sticky='W', padx=61)
        f4_color.grid(row=0, column=5, sticky='N', padx=0)

        # 接着是f5区域
        label_x = tk.Label(f5, text='X')
        label_x.grid(row=0, column=0, ipadx=3, padx=206, pady=4)
        label_y = tk.Label(f5, text='Y')
        label_y.grid(row=0, column=1, ipadx=3)

        self.in_list = lambda: in_data(text1, f3)
        self.train_list = lambda: train_data(text2, f4)
        self.test_list = lambda: test_data(text3, f5)

        ##### 函数区域################

        def file_op():
            path = filedialog.askopenfilenames()
            window.lift()
            return path

        def in_data(text1, f3):
            canvas_x = tk.Canvas(f3, bg='white', width=120, height=130)
            canvas_y = tk.Canvas(f3, bg='white', width=120, height=130)
            canvas_x.grid(row=1, column=0, sticky='N')
            canvas_x.pack_propagate(False)
            canvas_y.grid(row=1, column=1, sticky='N')
            canvas_y.pack_propagate(False)
            path = file_op()

            print("初始化路径:", path)
            columns_in = pd.read_csv(path[0]).columns
            text1.delete('1.0', 'end')
            for i in path:

                text1.insert('end', i + "\n")

            list1, list1_bt = gen_columns(columns_in, canvas_x)
            list2, list2_bt = gen_columns(columns_in, canvas_y)

            gen_scrollbar(canvas_x)
            gen_scrollbar(canvas_y)

            # 顺便添加all 按钮
            x_var = tk.IntVar()
            y_var = tk.IntVar()
            cx_all = tk.Checkbutton(f3, variable=x_var, text='all')
            cx_all.grid(row=2, column=0, sticky='N', pady=4)
            cy_all = tk.Checkbutton(f3, variable=y_var, text='all')
            cy_all.grid(row=2, column=1, sticky='N', pady=4)
            cx_all.config(command=lambda master=cx_all, list=list1: ppp(x_var, list))
            cy_all.config(command=lambda master=cy_all, list=list2: ppp(y_var, list))

            # 添加confirm 按钮
            def check():
                feature, label = [], []
                for i, h in zip(list1, list1_bt):
                    if i.get() == 0:
                        pass
                    if i.get() == 1:
                        c = h.cget('text')
                        feature.append(c)
                for i, h in zip(list2, list2_bt):
                    if i.get() == 0:
                        pass
                    if i.get() == 1:
                        c = h.cget('text')
                        label.append(c)
                # X, y
                self.feature_columns = feature
                self.label_columns = label
                in_data = pd.concat([pd.read_csv(i) for i in path], axis=0, ignore_index=True)
                #in_data.drop(in_data[in_data.iloc[:,104] > 15].index, inplace=True)
                in_data.index = ['origin' for i in range(in_data.shape[0])]
                in_data.dropna(inplace=True)
                X_lab = in_data[self.feature_columns]
                y_lab = in_data[self.label_columns]
                self.X_lab, self.y_lab = X_lab, y_lab
                print("训练集数据是:", self.X_lab, self.y_lab)
                self.label_exist.set(True)

            confirm = tk.Button(f3, text="confirm", command=lambda: check())
            confirm.place(x=830, y=0)

        def train_data(text2, f4):
            canvas_x = tk.Canvas(f4, bg='white', width=120, height=130)
            canvas_y = tk.Canvas(f4, bg='white', width=120, height=130)
            canvas_x.grid(row=1, column=0, sticky='N')
            canvas_x.pack_propagate(False)
            canvas_y.grid(row=1, column=1, sticky='N')
            canvas_y.pack_propagate(False)
            path = file_op()

            self.unlabel_pool = [str(os.path.basename(i)) for i in path]

            columns_in = pd.read_csv(path[0]).columns

            canvas_start = tk.Canvas(f4, bg='white', width=120, height=130)
            canvas_end = tk.Canvas(f4, bg='white', width=120, height=130)
            canvas_video = tk.Canvas(f4, bg='white', width=120, height=130)
            canvas_identity = tk.Canvas(f4, bg='white', width=120, height=130)

            canvas_start.grid(row=1, column=2, sticky='N')
            canvas_end.grid(row=1, column=3, sticky='N')
            canvas_video.grid(row=1, column=4, sticky='N')
            canvas_identity.grid(row=1, column=5, sticky='N')

            canvas_start.pack_propagate(False)
            canvas_end.pack_propagate(False)
            canvas_video.pack_propagate(False)
            canvas_identity.pack_propagate(False)
            # 生成列表:
            list1, list1_bt = gen_columns(columns_in, canvas_x)
            list2, list2_bt = gen_columns(columns_in, canvas_y)
            list3, list3_bt = gen_columns(columns_in, canvas_start)
            list4, list4_bt = gen_columns(columns_in, canvas_end)
            list5, list5_bt = gen_columns(columns_in, canvas_video)
            list6, list6_bt = gen_columns(columns_in, canvas_identity)

            gen_scrollbar(canvas_x)
            gen_scrollbar(canvas_y)
            gen_scrollbar(canvas_start)
            gen_scrollbar(canvas_end)
            gen_scrollbar(canvas_video)
            gen_scrollbar(canvas_identity)
            text2.delete('1.0', 'end')
            for i in path:

                text2.insert('end', i + "\n")
            # 顺便添加all 按钮
            x_var = tk.IntVar()
            y_var = tk.IntVar()
            start_var = tk.IntVar()
            end_var = tk.IntVar()
            video_var = tk.IntVar()
            identity_var = tk.IntVar()

            cx_all = tk.Checkbutton(f4, variable=x_var, text='all')
            cx_all.grid(row=2, column=0, sticky='N', pady=4)

            cy_all = tk.Checkbutton(f4, variable=y_var, text='all')
            cy_all.grid(row=2, column=1, sticky='N', pady=4)

            cstart_all = tk.Checkbutton(f4, variable=start_var, text='all')
            cstart_all.grid(row=2, column=2, sticky='N', pady=4)

            cend_all = tk.Checkbutton(f4, variable=end_var, text='all')
            cend_all.grid(row=2, column=3, sticky='N', pady=4)

            cvideo_all = tk.Checkbutton(f4, variable=video_var, text='all')
            cvideo_all.grid(row=2, column=4, sticky='N', pady=4)

            cidentity_all = tk.Checkbutton(f4, variable=identity_var, text='all')
            cidentity_all.grid(row=2, column=5, sticky='N', pady=4)

            cx_all.config(command=lambda master=cx_all, list=list1: ppp(x_var, list))
            cy_all.config(command=lambda master=cy_all, list=list2: ppp(y_var, list))
            cstart_all.config(command=lambda master=cy_all, list=list3: ppp(start_var, list))
            cend_all.config(command=lambda master=cy_all, list=list4: ppp(end_var, list))
            cvideo_all.config(command=lambda master=cy_all, list=list5: ppp(video_var, list))
            cidentity_all.config(command=lambda master=cy_all, list=list6: ppp(identity_var, list))

            confirm = tk.Button(f4, text="confirm", command=lambda: check())
            confirm.place(x=830, y=0)

            def check():
                feature, label, start, end, video, identity = [], [], [], [], [], []
                for i, h in zip(list1, list1_bt):
                    if i.get() == 0:
                        pass
                    if i.get() == 1:
                        c = h.cget('text')
                        feature.append(c)

                for i, h in zip(list2, list2_bt):
                    if i.get() == 0:
                        pass
                    if i.get() == 1:
                        c = h.cget('text')
                        label.append(c)

                for i, h in zip(list3, list3_bt):
                    if i.get() == 0:
                        pass
                    if i.get() == 1:
                        c = h.cget('text')
                        start.append(c)

                for i, h in zip(list4, list4_bt):
                    if i.get() == 0:
                        pass
                    if i.get() == 1:
                        c = h.cget('text')
                        end.append(c)

                for i, h in zip(list5, list5_bt):
                    if i.get() == 0:
                        pass
                    if i.get() == 1:
                        c = h.cget('text')
                        video.append(c)

                # for i in path:
                #     # import os
                #     # name =os.path.basename(i)
                #     # for h in range(pd.read_csv(i).shape[0]):
                #     #     identity.append(get_color(name))
                for i, h in zip(list6, list6_bt):
                    if i.get()==0:
                        pass
                    if i.get()==1:
                        c = h.cget('text')
                        identity.append(c)
                # 去除重复的从lab set
                # if self.label_exist == True:
                # X, y
                feature_columns = feature
                label_columns = label
                in_data = pd.concat([pd.read_csv(i) for i in path], axis=0, ignore_index=True)
                #in_data.drop(in_data[in_data['target'] > 15].index, inplace=True)


                if not pd.DataFrame(self.X_lab).empty:
                    print(in_data.shape)
                    print(np.where(in_data.iloc[:,0].isin(self.X_lab.iloc[:,0])))
                    in_data.drop(np.where(in_data.iloc[:,0].isin(self.X_lab.iloc[:,0]))[0],axis=0, inplace =True)
                    print(in_data.iloc[:,0])
                    print(self.X_lab.iloc[:,0])
                    print(self.X_lab.shape)
                    print(in_data.shape)

                in_data.dropna(axis=0, inplace=True)
                print(in_data)
                self.X_unlab, self.y_unlab, self.start_name, self.end_name, self.video_name, self.identity = in_data[feature_columns], in_data[label_columns], in_data[start], in_data[end], in_data[video], in_data[identity]
                if self.X_lab.columns.all() != self.X_unlab.columns.all():
                    window = tk.Toplevel()
                    window.geometry("400x200+200+150")
                    self.v1 = tk.IntVar()


                    def ss(value):
                        print(value.get())
                        if value.get() == 0:
                            self.X_unlab.columns = self.X_lab.columns
                            self.y_unlab.columns = self.y_lab.columns
                            print(self.y_unlab.columns)
                        if value.get() == 1:
                            self.X_lab.columns = self.X_unlab.columns
                            self.y_lab.columns = self.y_unlab.columns
                            print(self.y_lab.columns)

                    a = tk.LabelFrame(window, text="Checked name inconsistency in two dataset").grid()
                    Register_Label(window,
                                   text='initial pool\n' + str(self.X_lab.columns.tolist()[:4]) + '...').grid(row=0,
                                                                                                              column=0)
                    tk.Radiobutton(window, text='', variable=self.v1 , value=0 ).grid(row=1,
                                                                                                           column=0)
                    Register_Label(window,
                                   text='unlabel pool\n' + str(self.X_unlab.columns.tolist()[:4]) + "...").grid(
                        row=0, column=1)
                    tk.Radiobutton(window, text='', variable=self.v1, value=1).grid(row=1,
                                                                                                           column=1)
                    Click_Button(master=window, width=4, text='confirm', command=lambda: ss(self.v1 )).grid(row=1,
                                                                                                           column=2)



                 # 如果只有的化，那么就不用弄了
                # else:
                #     feature_columns = feature
                #     label_columns = label
                #     in_data = pd.concat([pd.read_csv(i) for i in path], axis=0, ignore_index=True)
                #     in_data.dropna(axis=0, inplace=True)
                #     self.X_unlab, self.y_unlab, self.start_name, self.end_name, self.video_name, self.identity = in_data[feature_columns], in_data[label_columns], in_data[start], in_data[end], in_data[video], in_data[identity]
                #
                #
                #     self.X_lab = pd.DataFrame(columns= feature_columns)
                #     self.y_lab = pd.DataFrame(columns= label_columns)
                #
                #     self.ind_collection = in_data.index
                #     print(self.ind_collection)




                # for i in [self.unlab_X, self.unlab_y, self.start, self.end, self.video, self.identity]:
                #     i.to_csv(path='./')
                #pd.DataFrame[feature, label, start, end, video]


        def test_data(text3, f5):
            canvas_x = tk.Canvas(f5, bg='white', width=120, height=130)
            canvas_y = tk.Canvas(f5, bg='white', width=120, height=130)
            canvas_x.grid(row=1, column=0, sticky='N')
            canvas_x.pack_propagate(False)
            canvas_y.grid(row=1, column=1, sticky='N')
            canvas_y.pack_propagate(False)
            path = file_op()

            self.test_pool = [str(os.path.basename(i)) for i in path]
            print("初始化路径:", path)
            columns_in = pd.read_csv(path[0]).columns
            text3.delete('1.0', 'end')
            for i in path:
                text3.insert('end', i + "\n")
            list1, list1_bt = gen_columns(columns_in, canvas_x)
            list2, list2_bt = gen_columns(columns_in, canvas_y)
            gen_scrollbar(canvas_x)
            gen_scrollbar(canvas_y)
            # 顺便添加all 按钮
            x_var = tk.IntVar()
            y_var = tk.IntVar()
            cx_all = tk.Checkbutton(f5, variable=x_var, text='all')
            cx_all.grid(row=2, column=0, sticky='N', pady=4)
            cy_all = tk.Checkbutton(f5, variable=y_var, text='all')
            cy_all.grid(row=2, column=1, sticky='N', pady=4)
            cx_all.config(command=lambda master=cx_all, list=list1: ppp(x_var, list))
            cy_all.config(command=lambda master=cy_all, list=list2: ppp(y_var, list))

            # 添加confirm 按钮
            confirm = tk.Button(f5, text="confirm", command=lambda: check(feature_columns=None, label_columns=None))
            confirm.place(x=830, y=0)

            def check(feature_columns=None, label_columns=None):
                feature, label = [], []
                for i, h in zip(list1, list1_bt):
                    if i.get() == 0:
                        pass
                    if i.get() == 1:
                        c = h.cget('text')
                        feature.append(c)

                for i, h in zip(list2, list2_bt):
                    if i.get() == 0:
                        pass
                    if i.get() == 1:
                        c = h.cget('text')
                        label.append(c)
                # X, y
                feature_columns = feature
                label_columns = label
                in_data = pd.concat([pd.read_csv(i) for i in path], axis=0, ignore_index=True)

                # if feature_columns != in_data.columns.all():
                #     window = tk.Toplevel()
                #     window.geometry("400x200+200+150")
                #     self.v1 = tk.IntVar()
                #
                #
                #     def ss(value):
                #         print(value.get())
                #         if value.get() == 0:
                #             in_data.iloc[:-1].columns = self.X_lab.columns
                #             in_data.iloc[-1].columns = self.y_lab.columns
                #
                #         if value.get() == 1:
                #             self.X_lab.columns = feature_columns
                #             self.y_lab.columns = label_columns
                #
                #
                #     a = tk.LabelFrame(window, text="Checked name inconsistency in two dataset").grid()
                #     Register_Label(window,
                #                    text='initial pool\n' + str(self.X_lab.columns.tolist()[:4]) + '...').grid(row=0,
                #                                                                                               column=0, sticky='w')
                #     tk.Radiobutton(window, text='', variable=self.v1 , value=0 ).grid(row=1,
                #                                                                                            column=0)
                #     Register_Label(window,
                #                    text='test pool\n' + str(in_data.columns.tolist()[:4]) + "...").grid(
                #         row=0, column=1, sticky='e')
                #     tk.Radiobutton(window, text='', variable=self.v1, value=1).grid(row=1,
                #                                                                                            column=1)
                #     Click_Button(master=window, text='confirm', command=lambda: ss(self.v1 )).grid(row=1,
                #                                                                                            column=2)

                print("xiangtong de biaoji  shi :",np.where(in_data.iloc[:, 0].isin(self.X_lab.iloc[:, 0]))[0])
                print("xiangtong de biaoji  shi :",
                      np.where(in_data.iloc[:, 0].isin(self.X_lab.iloc[:, 0]).any()))

                in_data.drop(np.where(in_data.iloc[:, 0].isin(self.X_lab.iloc[:, 0]))[0], axis=0, inplace=True)
                #in_data.drop(in_data[in_data['target'] > 15].index, inplace=True)
                print("对比的特征是", in_data.iloc[:, 0], self.X_lab.iloc[:, 0])


                # in_data.drop(np.where(in_data.iloc[:, 0].isin(self.X_lab.iloc[:, 0]))[0], axis=0, inplace=True)
                X_test = in_data[feature_columns]
                y_test = in_data[label_columns]
                print(X_test, y_test)
                self.X_test, self.y_test = X_test, y_test
                print("测试集数据是:\n",self.X_test, self.y_test)
                # for i in [X_test, y_test]:
                #     i.to_csv(path_or_buf='./')


        def _scroll(event, widget):
            widget.yview_scroll(int(-1 * (event.delta / 120)), "units")
        def gen_columns(columns_in, master):
            value_list, button_list = [], []
            for count, i in enumerate(columns_in):
                value = tk.IntVar()
                c = tk.Checkbutton(master, variable=value, text=i, bg='white')
                master.create_window(0, 22 * count, anchor='w', window=c)
                value_list.append(value)
                button_list.append(c)

            return value_list, button_list
        def ppp(master, list):
            if master.get() == 0:
                for i in list:
                    i.set(0)
            if master.get() == 1:
                for i in list:
                    i.set(1)
        def gen_scrollbar(master):
            # 滚动条必须后生成 否则会出错
            scroll_x = tk.Scrollbar(master, orient='vertical', width=14)
            scroll_x.pack(side='right', fill='y')
            master.config(yscrollcommand=scroll_x.set, scrollregion=master.bbox('all'))
            master.bind("<MouseWheel>", lambda event, widget=master: _scroll(event, master))
            scroll_x.config(command=master.yview)
    def ccc(self, master):
        root1 =  tk.Toplevel(master)
        root1.geometry("500x500+250+180")
        apps = app2(master=root1)

    # Categories Settings



    def eee(self, master):
        root = tk.Toplevel(master)
        root.geometry('730x500+250+180')
        root.title('regitser pannel')
        root.resizable(width=False,height=False)

        # f1 = tk.Frame(root, width=100, height=100, bg='red')
        # f1.grid()
        # f1.grid_propagate(False)
        str = 'Welcome to Register Panel, ' \
              '\nbelow list all info of files in Tree-view, including: ' \
              '\n1. Files Name.' \
              '\n2. File Type:' \
              '\n     a. Video' \
              '\n     b. Training(Sensor Data)' \
              '\n     c. Unlabel(Sensor Data)' \
              '\n     d. Testing(Sensor Data)' \
              '\n3. Start(Video)' \
              '\n4. Abs path(Absolute path)' \
              '\n' \
              '\nWhich you can double click row and make change'

        str2 = "Welcome to Register Panel, \ndouble click the row, make info change."

        hint = ttk.LabelFrame(root, text='Column info')
        hint.grid(padx=100)#padx=20, pady=20, sticky='nwse'
        tk.Label(hint, text=str2).grid(row=0,padx=100,sticky='WE')


        self.regi_data = pd.DataFrame({'file':[1], 'type':[2], 'Start':[3], 'Abs path':[4]})



        file = ttk.Button(root, text='Add', command= self.add)

        file_tree = Register_tree(root )
        file_tree.grid(pady=30)
        file_tree.set_(pd_form=self.regi_data)

        file_tree.pack_propagate(False)
        file_tree['show'] = 'headings'


    def add(self):
        paths = filedialog.askopenfilenames()
        f_lis=[]
        for i in paths:
            f_lis.append(os.path.basename(i))
        for count, i in enumerate(f_lis):
            pd.DataFrame({'file':i, 'type':None, 'Start':None, 'Abs path':paths[count]})







    def speed(self, speed):
        self.time_speed=speed
    def speed_get(self):
        print("现在的速度是:",self.time_speed)
        return self.time_speed

    def batch_set(self, num):
        self.batch_size = num
    def batch_get(self):
        return self.batch_size
    def thresh_set(self, num):
        self.thresh_hold = num
        self.var_thresh.set(num)
    def thresh_get(self):
        return self.thresh_hold

    def filter_set(self, num):
        self.filter_confusing = num
        self.var_confu.set(num)
    def filter_get(self):
        return self.filter_confusing


class File_Menu(tk.Menu):
    def __int__(self, master):
        super().__init__(master=master, tearoff=0)
class Speed_Menu(tk.Menu):
    def __int__(self, master=None, tearoff=0):
        super().__init__(master= master, tearoff=tearoff)
class Batch_Menu(tk.Menu):
    def __int__(self, master=None, tearoff=0):
        super().__init__(master= master, tearoff=tearoff)

class Mode_Menu(tk.Menu):
    def __int__(self, master=None, tearoff=0):
        super().__init__(master= master, tearoff=tearoff)

class Thresh_Menu(tk.Menu):
    def __int__(self, master=None, tearoff=0):
        super().__init__(master= master, tearoff=tearoff)

class Video_Canvas(tk.Canvas):
    def __init__(self, master=None,  width='640', height='480', bg='yellow'):
        super().__init__(master, width=width, height=height, bg=bg)

class Click_Button(tk.Button):
    def __init__(self, master=None, text='', **args):
        super().__init__(master, text=text,textvariable= None, **args)

class Selection_Area(ttk.Combobox):
    s = ["0 :移動Move","1 : 摂食Eat","2 :飲水Drink","3 :羽繕いPreening","4 :身震いShivering","5 :頭かきHead scratch","6 :尾振りTail swing","7 :巣箱に乗るGet on the nest  box","8 :巣箱を降りるGet off the nest box",
                                       "9 :止まり木に乗る Get on the perch","10:止まり木を降りる Get off the perch","11:静止 Stop","12:休息 Rest","13:砂浴びDust bathing","14:探査Litter exploration","15:首振りHead swing","16:バランスTo keep balance","17:センサつつきPeck the sensor","18:伸びStretching","19:嘴とぎBeak sharpening","20:地面つつきPeck the ground",
                                       "21:きょろきょろLook around","22:つつき攻撃Attack another hens","23:巣箱つつきPeck the nest box","24:つつかれPecked","25:センサつつかれPecked the sensor","26:----------------------"]
    def __init__(self, master=None, values = s, width='28', state= 'readonly',  *args, **kwargs):
        super().__init__(master, values= values,width=width, state=state,  *args, **kwargs)


    def get_value(self):
        return self.current()

class Time_input():
    def __init__(self, master):
        self.parent= master
        a = {"HH": tk.StringVar(), "MIN": tk.StringVar(), "SEC": tk.StringVar(), "MSEC": tk.StringVar()}
        self.can =tk.Frame(self.parent)
        self.can.grid(row=3, column=1)
        self.e1 = Register_Label(self.parent, text='Start   ')
        self.e2 = tk.Entry(master=self.can, width=3, textvariable=a['HH'])
        self.e3 = tk.Label(master=self.can, text=" : ")
        self.e4 = tk.Entry(master=self.can, width=3, textvariable=a['MIN'])
        self.e5 = tk.Label(master=self.can, text=" : ")
        self.e6 = tk.Entry(master=self.can, width=3, textvariable=a['SEC'])
        self.e7 = tk.Label(master=self.can, text=" . ")
        self.e8 = tk.Entry(master=self.can, width=3, textvariable=a['MSEC'])

        self.e1.grid(row=3, column=0, sticky='w')
        self.e2.grid(row=3, column=1, sticky='w')
        self.e3.grid(row=3, column=2, sticky='w')
        self.e4.grid(row=3, column=3, sticky='w')
        self.e5.grid(row=3, column=4, sticky='w')
        self.e6.grid(row=3, column=5, sticky='w')
        self.e7.grid(row=3, column=6, sticky='w')
        self.e8.grid(row=3, column=7, sticky='w')

        self.e_lis = [self.e1, self.e2, self.e3, self.e4, self.e5, self.e6, self.e7, self.e8]
        self.time = a

class Type_Area(ttk.Combobox):
    s = ["train", "test", 'Unlabel', 'Video']
    def __init__(self, master=None, values=s, width='20', parent=None,state="readonly",):
        super().__init__(master, values=values, width=width,state=state)
        self.bind('<<ComboboxSelected>>', self.judge_type)
        self.parent = parent
        self.time = None

        self.tp=None

    def get_value(self):
        return self.current()
    def judge_type(self,event):
        print(self.get_value())
        if self.get_value() == 3:
            # Register_Label(self.parent, text='Start   ').grid(row=3, column=0)
            # a = {"HH": tk.StringVar(), "MIN": tk.StringVar(), "SEC": tk.StringVar(), "MSEC": tk.StringVar()}
            # tk.Entry(master=self.parent, width=3, textvariable=a['HH']).grid(row=3, column=1)
            # tk.Label(master=self.parent, text=" : ").grid(row=3, column=2)
            # tk.Entry(master=self.parent, width=3, textvariable=a['MIN']).grid(row=3, column=3)
            # tk.Label(master=self.parent, text=" : ").grid(row=3, column=4)
            # tk.Entry(master=self.parent, width=3, textvariable=a['SEC']).grid(row=3, column=5)
            # tk.Label(master=self.parent, text=" . ").grid(row=3, column=6)
            # tk.Entry(master=self.parent, width=3, textvariable=a['MSEC']).grid(row=3, column=7)
            # self.time = a

            self.tp = Time_input(self.parent)
            self.time = self.tp.time

        else:
            for e in self.tp.e_lis:
                e.destroy()

            self.time= None

class Text_Area(tk.Text):
    def __init__(self, master=None,  width=50,height='35',*args, **kwargs ):
        super().__init__(master, width=width, height=height, *args, **kwargs )
class Scale_Progress(tk.Scale):
    #input = av.open("F:\壁纸\ppt pic\AB\動画-002\\0403.wmv")
    #stream1 = input.streams.video[0]

    def __init__(self, master=None,  orient='horizonta', variable=None, length='635', stream=None):
        super().__init__(master, orient=orient, variable= variable, length=length)
        if stream != None:
            self.configure(to=int(stream.duration / 1000), from_=0)
            self.v = variable

class Info_Tree(ttk.Treeview):
    def __init__(self, master=None, call_backs=None):
        super().__init__(master)
        # self.columnconfigure(0, weight=1)
        # self.rowconfigure(0, weight=1)
        # define the heading
        self.bind('<<TreeviewSelect>>', self.on_)
        self.selection_id = None
        self.selection_state = False
        self.scrollbar = None
    def on_(self, *args):
        print(self.selection_state)
        try:
            self.selection_id = self.selection()[0]
            print(self.selection())
            self.selection_state = True
            print(self.selection_id, self.selection_state)
        except IndexError:
            print("已经完成")

    def set_(self, pd_form, num=2):
        s = pd_form.columns.tolist()
        s.insert(0,'ind')
        s = s[:num]
        self.configure(columns=s)
        self.column('#0', width=0, anchor=tk.W,stretch='no' )

        for count, item in enumerate(s):
            if count==0:
                self.heading(s[count], text=item)
                self.column(s[count], width =44, anchor=tk.W, stretch='no')
            else:
                # if item == 'color':
                #     self.heading(s[count], text=item)
                #     self.column(s[count], width=0,anchor=tk.CENTER, stretch='yes')
                # else:
                self.heading(s[count], text=item)
                self.column(s[count], anchor=tk.CENTER, stretch='yes')

        for index, i in pd_form.iterrows():
            v = [index]
            for x in i:
                v.append(self.font_(x))
            self.insert('', 'end', iid = str(index),values = v[:num])

        self.scrollbar = ttk.Scrollbar(self,orient= tk.VERTICAL, command=self.yview )
        self.configure(yscrollcommand=self.scrollbar.set)
        self.scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

    def font_(self,word):
        return word

# class Player(tkVideoPlayer.TkinterVideo):
#     def __init__(self, master=None,  call_backs=None, *args, **kwargs):
#         super().__init__(master)
#         self.set_size((640, 480))
#         self.load(r'F:/壁纸/ppt pic/AB/動画-002/0713.wmv')

class Register_tree(ttk.Treeview):
    def __init__(self, master=None, call_backs=None):
        super().__init__(master)
        # self.columnconfigure(0, weight=1)
        # self.rowconfigure(0, weight=1)
        # define the heading
        self.bind('<<TreeviewSelect>>', self.Single_)
        self.bind("<Double-1>", self.Double_)

        self.selection_id = None
        self.selection_state = False
        self.scrollbar = None

    def Double_(self, *args):
        window = tk.Toplevel()
        window.geometry('625x450+400+260')
        window.title("File form")
        lf = ttk.LabelFrame(window, text='file info', width=400,height=400)
        lf.grid(padx = 100)
        lf.grid_propagate(False)

        Register_Label(lf, text="File   ").grid(row=0, column=0, sticky='w')
        filenam = None
        Register_Label(lf, text="filenam").grid(row=0, column=1 )

        Register_Label(lf, text='path   ').grid(row=1, column=0)
        filepath = None
        Register_Label(lf, text="filepath").grid(row=1, column=1 )


        Register_Label(lf, text='type   ').grid(row=2, column=0)

        TA = Type_Area(lf, values=['Train','Test','Unlabel','Video'], parent=lf)
        TA.grid(row=2, column=1, pady=20)

        def return_Tinfo():
            type = ['Train','Test','Unlabel','Video']
            try:
                time = TA.time['HH'].get()+":"+TA.time['MIN'].get()+":"+TA.time['SEC'].get()+"."+TA.time['MSEC'].get()
                print("现在是:",type[TA.get_value()], time)
            except:
                pass

        Click_Button(window, text='o', command=return_Tinfo).grid(sticky='N')




        # File : os.path.basename()
        # Type : Train, Test, Unlabel, separator(-------) ,Video
        # Start:
        # Feature Selection
        # Abs path

    def Single_(self, *args):
        print(self.selection_state)
        try:
            self.selection_id = self.selection()[0]
            print(self.selection())
            self.selection_state = True
            print(self.selection_id, self.selection_state)
        except IndexError:
            print("已经完成")

    def set_(self, pd_form, num=-1):
        s = pd_form.columns.tolist()
        s.insert(0, 'ind')
        s = s[:num]
        self.configure(columns=s)
        self.column('#0', width=0, anchor=tk.W, stretch='no')

        for count, item in enumerate(s):
            if count == 0:
                self.heading(s[count], text=item)
                self.column(s[count], width=44, anchor=tk.W, stretch='no')
            else:
                # if item == 'color':
                #     self.heading(s[count], text=item)
                #     self.column(s[count], width=0,anchor=tk.CENTER, stretch='yes')
                # else:
                self.heading(s[count], text=item)
                self.column(s[count], anchor=tk.CENTER, stretch='yes')

        for index, i in pd_form.iterrows():
            v = [index]
            for x in i:
                v.append(self.font_(x))
            self.insert('', 'end', iid=str(index), values=v[:num])

        self.scrollbar = ttk.Scrollbar(self, orient=tk.VERTICAL, command=self.yview)
        self.configure(yscrollcommand=self.scrollbar.set)
        self.scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

    def font_(self, word):
        return word

class numeric_entry(ttk.Entry):
    def __init__(self, master=None,  *args, **kwargs):
        super().__init__(master)
        self.config(
            validate='focusout',
            validatecommand=(self.register(self._validate),
                             '%S'
                             ),
            invalidcommand=(self.register(self._on_invalid),'%v')
        )

        self.error = tk.StringVar()
    def _validate(self, char):
        valid = char.isdigit()
        print("验证的结果是:",valid)
    def _on_invalid(self):
        print("验证失败，请输入数字")


