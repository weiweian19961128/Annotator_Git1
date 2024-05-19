import json
import os
import threading
import tkinter as tk
import tkinter.filedialog
from tkinter import ttk
from datetime import datetime

import cv2


import av
import numpy as np
from . import widget as w
from PIL import Image, ImageTk
import time
from tabulate import tabulate
import pandas as pd
import seaborn as sns

from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from lightgbm import LGBMClassifier
import japanize_matplotlib
from matplotlib import pyplot as plt
# Set font for matplotlib and seaborn


sns.set_style('darkgrid')





class Annotate_Frame(tk.Frame):
    """input for widgets"""

    def Frame_3(self):
        # 视频区域
        self.frame0 = tk.Frame(self, width='300', height='380')
        self.frame0.grid(row=0, column=0, sticky='N')

        # 分隔区域
        self.sep = tk.Label(self, width='1', height='28', bg='#333333')
        self.sep.grid(row=0, column=2, sticky='N')
        # 注册信息区域
        self.frame = tk.Frame(self, width='300', height='380')
        self.frame.grid(row=0, column=3, sticky='N')
        # 右边结果区域
        self.frame2 = tk.Frame(self, width='300', height='380')
        self.frame2.grid(row=0, column=4, sticky='N')

        self.Text_Frame1 = tk.Frame(master=self.frame2)
        self.Text_Frame1.grid(row=1, column=0, sticky='WN')
        self.Tree_Frame1 = tk.Frame(master=self.frame2, width=353, height=459, bg='red')
        self.Tree_Frame1.grid(row=1, column=0, sticky='WN')
        self.Tree_Frame1.pack_propagate(False)

        self.Tree_Frame2 = tk.Frame(master=self.frame2, width=353, height=459, bg='red')
        self.Tree_Frame2.grid(row=1, column=0, sticky='WN')
        self.Tree_Frame2.pack_propagate(False)

        self.Tree_Frame3 = tk.Frame(master=self.frame2, width=353, height=459, bg='red')
        self.Tree_Frame3.grid(row=1, column=0, sticky='WN')
        self.Tree_Frame3.pack_propagate(False)

        # self.frame2.tkraise()
        return {"Video": self.frame0, "Register": self.frame, "Info": self.frame2, 'Text': self.Text_Frame1,
                'Tree': self.Tree_Frame1, 'Search_Tree': self.Tree_Frame2, 'No label': self.Tree_Frame3}

    def __init__(self, parent, model=None, *args, **kwargs):
        super().__init__(parent, *args, **kwargs)
        self.model = model
        self.load_video()
        self.load_categories()
        self.parent_for_pause = parent
        # input & stream(av video object), flag(judge to pause/play video), img2(pause_frame) 1/(33*1.6)
        # Frame 1: video play area
        # Frame 2: register area
        # Frame 3: information area
        self.frame_all = self.Frame_3()
        self.var = {"Text": tk.StringVar, "entry1": tk.StringVar, "entry2": tk.IntVar, "entry3": tk.StringVar,
                    'entry4': tk.StringVar, 'entry5': tk.StringVar}

        self.Video_Canvas = w.Video_Canvas(master=self.frame_all['Video'])
        self.Tscale = w.Scale_Progress(master=self.frame_all['Video'], variable=tk.IntVar())

        self.Tscale.bind('<Button-1>', func=self.scale_state1)
        self.Tscale.bind('<ButtonRelease-1>', func=self.scale_state2)

        self.play_btn = w.Click_Button(master=self.frame_all['Video'], text='▶', width='8',
                                       command=lambda: self.judge())

        self.back_btn = w.Click_Button(master=self.frame_all['Video'], text='-4s',
                                       command=lambda: self.back_video(2)).grid(row=2, column=0, sticky='NW')
        w.Click_Button(master=self.frame_all['Video'], text='skip', command=lambda: self.skip()).grid(row=2, column=0,
                                                                                                      sticky='E')
        self.Video_Canvas.grid(row=0, column=0)
        self.Tscale.grid(row=1, column=0)
        self.play_btn.grid(row=2, column=0, sticky='N')

        w.Register_Label(master=self.frame_all['Register'], text='Video').grid(row=0, column=0, sticky='N')
        self.entry1 = w.Register_Entry(master=self.frame_all['Register'])
        w.Register_Label(master=self.frame_all['Register'], text='Behavior').grid(row=2, column=0, sticky='N')

        self.entry2 = w.Selection_Area(master=self.frame_all['Register'],
                                       values=[key + "." + value for key, value in self.model.dict.items()])
        w.Register_Label(master=self.frame_all['Register'], text='Color').grid(row=4, column=0, sticky='N')
        self.entry3 = w.Register_Entry(master=self.frame_all['Register'])
        w.Register_Label(master=self.frame_all['Register'], text='Start Time').grid(row=6, column=0, sticky='N')
        self.entry4 = w.Register_Entry(master=self.frame_all['Register'])
        w.Register_Label(master=self.frame_all['Register'], text='End Time').grid(row=8, column=0, sticky='N')
        self.entry5 = w.Register_Entry(master=self.frame_all['Register'])
        w.Click_Button(master=self.frame_all['Register'], text='Label', command=lambda: self.train()).grid(row=10,
                                                                                                           column=0,
                                                                                                           sticky='W')
        w.Click_Button(master=self.frame_all['Register'], text='Auto', command=lambda: self.test_auto()).grid(row=10,
                                                                                                              column=0,
                                                                                                              sticky='S')

        w.Click_Button(master=self.frame_all['Register'], text='Cancel', command=lambda: self.cancel_step()).grid(
            row=10, column=0, sticky='E')

        self.entry1.grid(row=1, column=0, sticky='N')
        self.entry2.grid(row=3, column=0, sticky='N')
        self.entry3.grid(row=5, column=0, sticky='N')
        self.entry4.grid(row=7, column=0, sticky='N')
        self.entry5.grid(row=9, column=0, sticky='N')

        self.search_dic = [key + "." + value for key, value in self.model.dict.items()]
        self.search_dic.append("--------------Switch to Uncertainty Query--------------")
        self.search_entry = w.Selection_Area(master=self.frame_all['Info'], width='39', values=self.search_dic)
        self.search_entry.grid(row=0, column=0, sticky='W')
        w.Click_Button(master=self.frame_all['Info'], text='search', command=lambda: self.search()).grid(row=0,
                                                                                                         column=0,
                                                                                                         sticky='E')

        # Part 1 frame(Text)
        self.Text = w.Text_Area(master=self.frame_all['Text'])
        self.Text.grid(row=1, column=0)

        # Part 2 frame(Tree)
        self.Tree = w.Info_Tree(master=self.frame_all['Tree'])
        self.Tree.pack(fill='both', expand=True)
        self.search_Tree = w.Info_Tree(master=self.frame_all['Search_Tree'])
        self.search_Tree.pack(fill='both', expand=True)

        # Part 3 nolabel Tree
        self.no_label_Tree = w.Info_Tree(master=self.frame_all['No label'])
        self.no_label_Tree.pack(fill='both', expand=True)

        self.Change = w.Click_Button(master=self.frame_all['Info'], text='🌳', command=lambda: self.rais()).grid(row=1,
                                                                                                                column=1,
                                                                                              sticky='W')#query
        w.Click_Button(master=self.frame_all['Info'], text='⬅️', command=lambda: self.query()).grid(row=2,
                                                                                                         column=0,
                                                                                                         sticky='NW')
        w.Click_Button(master=self.frame_all['Info'], text='Pooling🛁', command=lambda: self.calculate()).grid(row=2,
                                                                                                             column=0,
                                                                                                             sticky='N')
        w.Click_Button(master=self.frame_all['Info'], text='Save📂', command=lambda: self.on()).grid(row=2, column=0,
                                                                                                   sticky='NE')

        self.model.Main_menu.file.add_command(label='🐔Class Setting', command=lambda: self.ddd(self))
        self.model.Main_menu.file.add_command(label='🎠Check point', command=lambda: self.o())
        self.model.Main_menu.analysis_menu.add_command(label='Acc and F1 test', command=lambda: self.s())

        self.input, self.stream, self.time_speed, self.flag, self.img2, self.times = None, None, 0.0188, 0, None, None
        self._paused = False

        self.count = 0
        self.present = 0
        self.frame_all['Text'].tkraise()

        # serve for category setting
        self.Name_ = tk.StringVar()
        self.class_name = tk.IntVar()
        self.dict = {}
        self.cat_setting = tk.StringVar()
        self.combo = None

    def load_video(self):

        try:
            with open("./Resume/fix_setting/video_path.json", 'r') as f:
                data = json.load(f)
                print(data, type(data))
                self.model.Main_menu.video_path = np.array([i for i in data.keys()])
                self.model.Main_menu.V_dict = data
        except:
            pass

        # p = pd.read_csv("./Resume/fix_setting/video_path.csv", header=None)
        # v = p.iloc[:, 1].values.tolist()
        # t = p.iloc[:, 2].values.tolist()
        # self.model.Main_menu.video_path = v
        # print(v, t)
        # print(type(v[0]), type(t[0]))
        #
        # self.model.Main_menu.V_dict = {i:h for i, h in zip(v, t)}
        print(self.model.Main_menu.video_path, self.model.Main_menu.V_dict)

    def load_categories(self):
        try:
            with open("./Resume/fix_setting/categories.json", 'r') as f:
                self.model.dict = json.load(f)

            if all(["Confusing(behaviors>2)" not in i for i in self.model.dict.values()]):
                size = len(self.model.dict)
                self.model.dict[str(size)] = "Confusing(behaviors>2)"

            if all(["Confusing(Not sure)" not in i for i in self.model.dict.values()]):
                size = len(self.model.dict)
                self.model.dict[str(size)] = "Confusing(Not sure)"
        except:
            pass

    def test_auto(self, k=5):
        self.model.semi_learning()
        # auto label
        # for i in range(990):
        #     self.model.auto_query()
        # self.model.stop_criterions(5)
        # save file
        self.model.write_stoprocess()

    def scale_state1(self, event):
        self.flag = 0

    def scale_state2(self, event):
        point = self.Tscale.get()

        # 会导致video的运行
        # self.movie.set(0, (self.point) * 1000)
        self.input.seek(offset=int(point * 1000), stream=self.stream, any_frame=False)

        i = next(self.input.decode(self.stream))
        self.orient = score = i.pts
        i.reformat(width=640, height=480)
        i = i.to_image()
        self.img2 = ImageTk.PhotoImage(i)
        self.Video_Canvas.create_image(0, 0, anchor='nw', image=self.img2)
        self.Tscale.set(int(score / 1000))
        self.Video_Canvas.update()
        # print(self.time_speed)
        time.sleep(self.time_speed)
        self.pause()

    def back_video(self, times):
        self.input.seek(offset=int((self.Tscale.get() - 2) * 1000), stream=self.stream, any_frame=False)
        # videocapture.set(0,value)

    def on(self):
        window = tk.Toplevel(self)
        window.geometry('500x500+600+300')
        window.title("class setting")

        self.selected = []

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

        def check():
            label = []
            for i, h in zip(list1, list1_bt):
                if i.get() == 0:
                    pass
                if i.get() == 1:
                    c = h.cget('text')
                    label.append(c)
            self.selected = {int(i): self.model.dict[str(i)] for i in label}

            list = [self.entry1, self.entry3, self.entry4, self.entry5]
            p = [i.get() for i in list]
            p.insert(1, self.entry2.current())

            path = tk.filedialog.askdirectory()
            self.model.form_data(entry_info=p, selection_labels=[self.selected, path])

            window.destroy()

        def gen_scrollbar(master):
            # 滚动条必须后生成 否则会出错
            scroll_x = tk.Scrollbar(master, orient='vertical', width=14)
            scroll_x.pack(side='right', fill='y')

            master.config(yscrollcommand=scroll_x.set, scrollregion=master.bbox('all'))
            master.bind("<MouseWheel>", lambda event, widget=master: _scroll(event, master))
            scroll_x.config(command=master.yview)

        # selection area
        w.Register_Label(window, text='label classes selection').grid(row=0, column=0, sticky='N')
        canvas_x = tk.Canvas(window, bg='white', width=320, height=130)
        canvas_x.pack_propagate(False)

        # catalogs
        list1, list1_bt = gen_columns(self.model.rf.classes_, canvas_x)

        # checkbutton
        x_var = tk.IntVar()
        cx_all = tk.Checkbutton(window, variable=x_var, text='all')
        cx_all.config(command=lambda master=cx_all, list=list1: ppp(x_var, list1))
        confirm = tk.Button(window, text="confirm", command=lambda: check())
        gen_scrollbar(canvas_x)

        canvas_x.grid(row=1, column=0, sticky='N', padx=60, pady=5)
        cx_all.grid(row=2, column=0, sticky='N', pady=4)
        confirm.grid(row=3, column=0)

    # Analysis
    def s(self):
        from sklearn.metrics import accuracy_score as Acc, f1_score
        from sklearn.ensemble import RandomForestClassifier as rf
        from matplotlib import pyplot as plt
        # plt.rcParams.update({"figure.dpi": 96})

        def Accuracy():
            score_ = []
            X, y = self.model.Main_menu.X_lab.copy(), self.model.Main_menu.y_lab.copy()
            y = pd.DataFrame(y)['target']
            X.index, y.index = range(X.shape[0]), range(y.shape[0])
            class_clean = y.value_counts().sort_index().index[:-2].values.tolist()
            print("所有的数据是:", y, type(y))
            handle = []
            for i in class_clean:
                if isinstance(i, tuple):
                    handle.append(i[0])
                else:
                    handle.append(i)
            class_clean = handle
            print("\n干净的类是:", class_clean)
            index_clean = y[y.isin(class_clean)].index
            print("\n干净的标签是:", index_clean)

            print("\n干净的数据集X是:", X.loc[index_clean])
            print("\n干净的数据集y是:", y.loc[index_clean])
            print("\n标记验证", X.loc[index_clean].index)
            print("\n标记验证", y.loc[index_clean].index)
            print("\n类统计:", y.loc[index_clean].value_counts())
            X_test, y_test = self.model.Main_menu.X_test, self.model.Main_menu.y_test



            model1 = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000))
            model2 = rf()
            model3 = LGBMClassifier()

            model_nam = ['Lgr', 'rf', 'LightGBM']
            clean = 'No Confusing' + " ({})".format(y.loc[index_clean].shape[0])
            confu = 'with Confusing' + " ({})".format(y.shape[0])

            def model_output(model):
                # 情况1， 干净数据集
                Clean_Confusing = {}
                nam = type(model).__name__
                print("模型的名称是:", nam)
                model.fit(X.loc[index_clean], y.loc[index_clean])
                pred = model.predict(X_test)

                Clean_Confusing[clean] = Acc(y_test, pred)

                # 情况2， 所有数据集
                model.fit(X, y)
                pred2 = model.predict(X_test)

                Clean_Confusing[confu] = Acc(y_test, pred2)
                print(Clean_Confusing)
                return Clean_Confusing

            score_.append(model_output(model1))
            score_.append(model_output(model2))
            score_.append(model_output(model3))
            df = pd.DataFrame(score_)
            df['model'] = model_nam
            df1 = df.melt(id_vars=['model'], value_vars=[clean, confu], var_name='Whether Confusing',
                          value_name='value')

            graph = sns.catplot(data=df1, x='model', y='value', hue='Whether Confusing',   kind='bar',
                                dodge=True, palette='YlGnBu', )
            graph.set_axis_labels("Model", "Accuracy")
            graph.set_titles("RandomForest")

            ax = graph.facet_axis(0, 0)

            for p in ax.patches:
                ax.text(p.get_x() + 0.015,
                        p.get_height() * 1.015,
                        '{0:.3f}'.format(p.get_height()),
                        color='blue', rotation='horizontal',  fontsize=7)



            plt.show()
            return graph

        def f1_W():
            score_ = []
            X, y = self.model.Main_menu.X_lab.copy(), self.model.Main_menu.y_lab.copy()
            y = pd.DataFrame(y)['target']
            X.index, y.index = range(X.shape[0]), range(y.shape[0])
            class_clean = y.value_counts().sort_index().index[:-2].values.tolist()
            print("所有的数据是:", y, type(y))
            handle = []
            for i in class_clean:
                if isinstance(i, tuple):
                    handle.append(i[0])
                else:
                    handle.append(i)
            class_clean = handle
            print("\n干净的类是:", class_clean)
            index_clean = y[y.isin(class_clean)].index
            print("\n干净的标签是:", index_clean)

            print("\n干净的数据集X是:", X.loc[index_clean])
            print("\n干净的数据集y是:", y.loc[index_clean])
            print("\n标记验证", X.loc[index_clean].index)
            print("\n标记验证", y.loc[index_clean].index)
            print("\n类统计:", y.loc[index_clean].value_counts())
            X_test, y_test = self.model.Main_menu.X_test, self.model.Main_menu.y_test
            model1 = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000))
            model2 = rf()
            model3 = LGBMClassifier()

            model_nam = ['Lgr', 'rf', 'LightGBM']
            clean = 'No Confusing' + " ({})".format(y.loc[index_clean].shape[0])
            confu = 'with Confusing' + " ({})".format(y.shape[0])
            clean = 'No Confusing' + " ({})".format(y.loc[index_clean].shape[0])
            confu = 'with Confusing' + " ({})".format(y.shape[0])

            def model_output(model):
                # 情况1， 干净数据集
                Clean_Confusing = {}
                nam = type(model).__name__
                print("模型的名称是:", nam)
                model.fit(X.loc[index_clean], y.loc[index_clean])
                pred = model.predict(X_test)

                Clean_Confusing[clean] = f1_score(y_test, pred, average="macro")

                # 情况2， 所有数据集
                model.fit(X, y)
                pred2 = model.predict(X_test)

                Clean_Confusing[confu] = f1_score(y_test, pred2, average="macro")
                return Clean_Confusing

            from matplotlib import pyplot as plt
            score_.append(model_output(model1))
            score_.append(model_output(model2))
            score_.append(model_output(model3))
            df = pd.DataFrame(score_)
            df['model'] = model_nam
            df1 = df.melt(id_vars=['model'], value_vars=[clean, confu],
                          var_name='Whether Confusing', value_name='value')

            graph = sns.catplot(data=df1, x='model', y='value', hue='Whether Confusing',
                                dodge=True, kind='bar', palette='YlGnBu')

            graph.set_axis_labels("Model", "f1-macro")
            graph.set_titles("RandomForest")

            ax = graph.facet_axis(0, 0)

            for p in ax.patches:
                ax.text(p.get_x() + 0.005,
                        p.get_height() * 1.015,
                        '{0:.3f}'.format(p.get_height()),
                        color='blue', rotation='horizontal',  fontsize=7)

            # for count, i in enumerate(df1['value'].values):
            #     graph.annotate(text=str(i), xy=(count,i))

            plt.show()
            return graph

        def Cat_confusion():
            score_ = []
            X, y = self.model.Main_menu.X_lab.copy(), self.model.Main_menu.y_lab.copy()
            y = pd.DataFrame(y)['target']
            X.index, y.index = range(X.shape[0]), range(y.shape[0])
            class_clean = y.value_counts().sort_index().index[:-2].values.tolist()
            print("所有的数据是:", y, type(y))
            handle = []
            for i in class_clean:
                if isinstance(i, tuple):
                    handle.append(i[0])
                else:
                    handle.append(i)
            class_clean = handle
            print("\n干净的类是:", class_clean)
            index_clean = y[y.isin(class_clean)].index
            print("\n干净的标签是:", index_clean)

            print("\n干净的数据集X是:", X.loc[index_clean])
            print("\n干净的数据集y是:", y.loc[index_clean])
            print("\n标记验证", X.loc[index_clean].index)
            print("\n标记验证", y.loc[index_clean].index)
            print("\n类统计:", y.loc[index_clean].value_counts())
            X_test, y_test = self.model.Main_menu.X_test, self.model.Main_menu.y_test
            model1 = rf(n_estimators=100, max_depth=12, n_jobs=-1)
            model1.fit(X, y)

            pred = model1.predict(X_test)

            from sklearn.metrics import confusion_matrix

            def plot_cm(y_true, y_pred, figsize=(20, 20)):
                cm = confusion_matrix(y_true, y_pred, labels=np.unique(y_true))
                cm_sum = np.sum(cm, axis=1, keepdims=True)
                cm_perc = cm / cm_sum.astype(float) * 100
                annot = np.empty_like(cm).astype(str)
                nrows, ncols = cm.shape
                for i in range(nrows):
                    for j in range(ncols):
                        c = cm[i, j]
                        p = cm_perc[i, j]
                        if i == j:
                            s = cm_sum[i]
                            annot[i, j] = '%.1f%%\n%d/%d' % (p, c, s)
                        elif c == 0:
                            annot[i, j] = ''
                        else:
                            annot[i, j] = '%.1f%%\n%d' % (p, c)
                cm = pd.DataFrame(cm, index=np.unique(y_true), columns=np.unique(y_true))
                cm.index.name = 'Actual'
                cm.columns.name = 'Predicted'
                fig, ax = plt.subplots(figsize=figsize)
                graph = sns.heatmap(cm, cmap="YlGnBu", annot=annot, fmt='', ax=ax)
                plt.title('Recall')
                plt.show()
                return graph.get_figure()

            a = plot_cm(y_test, pred)
            return a

        def label_counts():
            japanize_matplotlib.japanize()
            sns.set(font='IPAexGothic')
            score_ = []
            X, y = self.model.Main_menu.X_lab.copy(), self.model.Main_menu.y_lab.copy()
            y = pd.DataFrame(y)['target']
            X.index, y.index = range(X.shape[0]), range(y.shape[0])
            class_clean = y.value_counts().sort_index().index[:-2].values.tolist()
            print("所有的数据是:", y, type(y))
            handle = []
            for i in class_clean:
                if isinstance(i, tuple):
                    handle.append(i[0])
                else:
                    handle.append(i)
            class_clean = handle
            print("\n干净的类是:", class_clean)
            index_clean = y[y.isin(class_clean)].index
            print("\n干净的标签是:", index_clean)

            print("\n干净的数据集X是:", X.loc[index_clean])
            print("\n干净的数据集y是:", y.loc[index_clean])
            print("\n标记验证", X.loc[index_clean].index)
            print("\n标记验证", y.loc[index_clean].index)
            print("\n类统计:", y.loc[index_clean].value_counts())

            statistic = y.value_counts().to_frame().sort_index().reset_index()
            print(statistic)
            statistic['label'] = statistic['target'].apply(lambda x: np.array(self.model.dict[str(x)]))
            print(statistic)

            graph = sns.catplot(data=statistic, x='count', y='label' , kind='bar')
            ax = graph.facet_axis(0, 0)
            for p in ax.patches:
                width = p.get_width()
                ax.text(10 + p.get_width(), p.get_y() + 0.55 * p.get_height(),
                         str(int(width)),
                         ha='center', va='center', size='medium', color='blue')

            plt.title("Labeled Behaviors")
            plt.xlabel("Counts")



            plt.tight_layout()
            plt.show()
            return graph

        def pool():
            japanize_matplotlib.japanize()
            sns.set(font='IPAexGothic')
            X, y = self.model.Main_menu.X_lab.copy(), self.model.Main_menu.y_lab.copy()
            y = pd.DataFrame(y)['target']
            X.index, y.index = range(X.shape[0]), range(y.shape[0])
            class_clean = y.value_counts().sort_index().index[:-2].values.tolist()
            print("所有的数据是:", y, type(y))
            handle = []
            for i in class_clean:
                if isinstance(i, tuple):
                    handle.append(i[0])
                else:
                    handle.append(i)
            class_clean = handle
            print("\n干净的类是:", class_clean)
            index_clean = y[y.isin(class_clean)].index
            print("\n干净的标签是:", index_clean)

            print("\n干净的数据集X是:", X.loc[index_clean])
            print("\n干净的数据集y是:", y.loc[index_clean])
            print("\n标记验证", X.loc[index_clean].index)
            print("\n标记验证", y.loc[index_clean].index)
            print("\n类统计:", y.loc[index_clean].value_counts())

            X_unlab = self.model.Main_menu.X_unlab

            model1 = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000))
            model2 = rf()
            model3 = LGBMClassifier()

            model_nam = ['LogisticRegression', 'RandomForest', 'LightGBM']
            for m_nam, m in zip(model_nam,[model1, model2, model3]):
                m.fit(X, y)
                print(X_unlab)
                pred = m.predict(X_unlab)
                Ser = pd.Series(pred, name='target')
                statistic = Ser.value_counts().to_frame().sort_index().reset_index()
                print(statistic)
                statistic['label'] = statistic['target'].apply(lambda x: self.model.dict[str(x)])
                print(statistic)

                graph = sns.catplot(data=statistic, y='label', x='count' , kind='bar')
                ax = graph.facet_axis(0, 0)
                plt.title("Unlabeled Behaviors-"+m_nam)
                plt.xlabel("Counts")



                # for p in ax.patches:
                #     ax.text(p.get_x()* 1.02 ,
                #             p.get_height()+ 0.015 ,
                #             '{0:.2f}'.format(p.get_x()),
                #             color='black', rotation='horizontal', size='medium')

                for p in ax.patches:
                    width = p.get_width()
                    plt.text(10 + p.get_width(), p.get_y() + 0.55 * p.get_height(),
                             str(int(width)),
                             ha='center', va='center', size='medium', color='blue')
                plt.tight_layout()

            plt.show()
            return graph

        def lab_vs_unlab():
            t1 = threading.Thread(target=label_counts())
            t2 = threading.Thread(target=pool())
            t1.start()
            t2.start()

        def save_graphs():
            plt1 = Accuracy()
            plt2 = f1_W()
            plt3 = Cat_confusion()
            plt4 = label_counts()
            plt5 = pool()

            path = tkinter.filedialog.askdirectory()
            plt1.savefig(path + "/acc.png", dpi=200)
            plt2.savefig(path + "/f1_weight.png", dpi=200)
            plt3.savefig(path + "/confusion_matrix.png", dpi=200)
            plt4.savefig(path + "/label.png", dpi=200)
            plt5.savefig(path + "/predicted unlabel.png", dpi=200)

        # a = threading.Thread(target=self._main_func)
        # # score = self.model.stop_criterion()
        # # print(score)
        window = tk.Toplevel(self)
        window.geometry('600x300+300+260')
        window.title("Analysis Tool")

        lf_performance = ttk.LabelFrame(window, text='Model Performance Analysis')
        lf_behavior = ttk.LabelFrame(window, text='Behavior Counts')

        lf_performance.grid(row=0, column=0, pady=5, padx=15, sticky='NW')
        lf_behavior.grid(row=0, column=1, sticky='NW')
        ttk.Button(window, text='save', command=lambda: save_graphs()).grid(row=0, column=2, padx=10, sticky='SE')

        accuracy_ = ttk.Button(lf_performance, text='Accuracy',
                               command=lambda: Accuracy())  # threading.Thread(target =Accuracy))
        f1_weigthed = ttk.Button(lf_performance, text='f1-macro', command=lambda: f1_W())
        confusing = ttk.Button(lf_performance, text='Confusion Matrix', command=lambda: Cat_confusion())

        Category = ttk.Button(lf_behavior, text='Labeled behaviors', command=lambda: label_counts())
        Potential_unlab = ttk.Button(lf_behavior, text='Pool Predicted', command=lambda: pool())
        Comparison_ = ttk.Button(lf_behavior, text='lab vs Unlab', command=lambda: lab_vs_unlab())

        accuracy_.grid(sticky='nw')
        f1_weigthed.grid(sticky='nw')
        confusing.grid(sticky='nw')

        Category.grid(sticky='nw')
        Potential_unlab.grid(sticky='nw')
        Comparison_.grid(sticky='nw')

        # self.model.draw_()

    # resume previous work
    def o(self):
        video, lab, identity, start, end = self.model.resume1(entry_current=self.entry2.current())
        self.entry1.delete(0, 'end')
        self.entry1.insert('end', video)
        self.entry3.delete(0, 'end')
        self.entry3.insert('end', identity)
        self.entry4.delete(0, 'end')
        self.entry4.insert('end', start)
        self.entry5.delete(0, 'end')
        self.entry5.insert('end', end)
        self.skip()
        self.entry2.current(lab)
        self.Text.insert('end', '\nLast work load successfully,\nPlease click Refresh button!!!')

    # Setting categories
    def ddd(self, master):
        window = tk.Toplevel(master)
        window.geometry('600x300+300+260')
        window.title("feature")
        # 三列
        lf = ttk.LabelFrame(window, text='please set class without blank')
        lf.grid(padx=20, pady=20)
        # ttk.Spinbox(window, from_=0, to_= 1000, textvariable= self.Series).grid(row=0, column=0, sticky=tk.W)

        ttk.Label(lf, text='Class Label (Number Only)').grid(row=1, column=0, sticky=tk.W)
        ttk.Entry(lf, width=18, textvariable=self.class_name).grid(row=2, column=0, sticky=tk.W)

        ttk.Label(lf, text='Name Description').grid(row=3, column=0, sticky=tk.W)
        ttk.Entry(lf, width='28', textvariable=self.Name_).grid(row=4, column=0, sticky=tk.W)

        self.combo = ttk.Combobox(lf, width='28', textvariable=self.cat_setting, state="readonly")
        self.combo.grid(row=0, column=2, sticky=tk.W)
        ttk.Button(lf, text='-', width=2, command=lambda: self.del_category()).grid(row=0, column=3)

        ttk.Button(lf, text='default', command=lambda: self.default_category()).grid(row=5, column=3)

        self.bt = ttk.Button(lf, text='+', width=2, command=lambda: self.add_category())
        self.bt.grid(row=5, column=0, sticky=tk.E)

        self.bt2 = ttk.Button(window, text='confirm', command=lambda: self.cate_confirm())
        self.bt2.grid(row=1, column=0, sticky=tk.E)

    def del_category(self):
        num = self.combo.current()
        options = list(self.combo['values'])
        if bool(options) == True:
            del (options[num])
            self.combo['values'] = options

            try:
                self.dict.pop(str(num))
            except:
                pass
            if num > 0:
                self.combo.current(num - 1)
        else:
            self.combo.set("")

    def default_category(self):

        with open("./Resume/fix_setting/Default.json", 'r') as f:
            self.dict = json.load(f)
            self.combo.configure(values=[key + "." + value for key, value in self.dict.items()])

    def add_category(self):
        from math import isnan
        if len(list(self.combo['values'])) == 0:
            self.dict = {}

        key = int(self.class_name.get())
        value = self.Name_.get()
        print(self.Name_.get())
        # if key not in self.dict.keys() and bool(self.dict):
        #     print("添加成功，没有在添加的列表里")
        self.dict[key] = str(value)
        print(self.dict)
        clean_dic = {}
        for k, value in self.dict.items():
            if not isnan(int(k)):
                clean_dic[int(k)] = value
        list_ = sorted(list(clean_dic.keys()))

        clean_dic = {str(k): clean_dic[k] for k in list_}
        # v = [str(h) for i, h in clean_dic.items()] #str(i) + ".  " +
        v = [key + "." + value for key, value in clean_dic.items()]
        judge_transition = []
        judge_context = []
        self.dict = clean_dic
        self.combo.configure(values=v)

    def cate_confirm(self):
        if all(['transition' not in i.lower() for i in list(self.combo['values'])]):
            option = list(self.combo['values'])
            option.append(str(len(option)) + ".  Transition Confusing Labels")
            self.combo['values'] = option
        if all(['context' not in i.lower() for i in list(self.combo['values'])]):
            option = list(self.combo['values'])
            option.append(str(len(option)) + ".  Context Confusing Labels")
            self.combo['values'] = option
        with open('./Resume/fix_setting/categories.json', 'w') as f:
            # f.write(str(list(self.combo['values'])))
            json.dump(self.dict, f)
        self.load_categories()
        self.entry2.configure(values=[key + "." + value for key, value in self.model.dict.items()])

        self.search_dic = [key + "." + value for key, value in self.model.dict.items()]
        self.search_dic.append("--------------Switch to Uncertainty Query--------------")
        self.search_entry.configure(values=self.search_dic)

        self.dict = self.model.dict

    def calculate(self):
        # if self.model.Main_menu.label_exist.get() == True:
        # margin, collection, model = self.model.margin_rank(self.model.rf1() , self.model.Main_menu.X_lab, self.model.Main_menu.y_lab, self.model.Main_menu.X_unlab)

        print(f"哪个  是{self.model.Main_menu.filter_confusing}")
        if self.model.Main_menu.filter_confusing == 1:
            confu_idx = np.where(self.model.rf.predict(self.model.Main_menu.X_unlab) >= len(self.model.rf.classes_) - 2)[0]
            confu_idx = self.model.Main_menu.X_unlab.iloc[confu_idx, :].index

            self.model.Main_menu.X_unlab = self.model.Main_menu.X_unlab.drop(index=confu_idx)




        margin, collection, model = self.model.Active_Learning()
        self.Text.delete('1.0', 'end')
        self.Text.insert('end', margin)
        self.Text.insert('end', '\n')

        # l: AL progress, current progress(%).
        percentage = self.model.Main_menu.X_lab.shape[0] / 3400
        formatted_percentage = f"{percentage:.2%}"

        l = pd.DataFrame([self.model.Main_menu.X_lab.shape[0], self.model.Main_menu.X_unlab.shape[0],
                          formatted_percentage], index=['lab pool:', 'unlab pool:', 'progress:'])
        self.Text.insert('end', tabulate(l, tablefmt='simple'))

        import time

        # Record the start time
        start_time = time.time()
        if self.model.delete_tree == True:
            for i in self.Tree.get_children():
                self.Tree.delete(i)
            if self.Tree.scrollbar:
                self.Tree.scrollbar.destroy()
            self.Tree.set_(self.model.tree_form(model))

        # 如果改变则有scroll bar
        if self.search_Tree.scrollbar:
            for h in self.search_Tree.get_children():
                self.search_Tree.delete(h)
            self.search_Tree.scrollbar.destroy()
            self.search_Tree.set_(self.model.tree_form(model, inds_=self.model.search_index))
        self.model.delete_tree = False

        # Record the end time
        end_time = time.time()

        # Calculate the elapsed time
        elapsed_time = end_time - start_time

        #print(f"Elapsed time: {elapsed_time} seconds")
        self.Text.insert('end', f"\n{self.model.Main_menu.estimator_} model fit-time: {elapsed_time} seconds")

        # else:
        #     print(self.model.Main_menu.ind_collection[0])
        #     if self.no_label_Tree.scrollbar:
        #         for h in self.no_label_Tree.get_children():
        #             self.no_label_Tree.delete(h)
        #         self.no_label_Tree.scrollbar.destroy()
        #         self.no_label_Tree.set_(self.model.no_label_tree(), num=4)
        #     else:
        #         self.no_label_Tree.set_(self.model.no_label_tree(), num=4)

    def query(self):
        # if self.model.Main_menu.label_exist.get() == True:
        if self.search_entry.current() == -1 or self.search_entry.current() == len(self.search_dic) - 1:
            video, identity, start, end, ind, state = self.model.query_label(search_state=self.search_entry.current(),
                                                                             select_ind=self.Tree.selection_id,
                                                                             selection_state=self.Tree.selection_state)
            try:
                self.Tree.selection_state = False
                self.Tree.delete(ind)
                self.search_Tree.delete(ind)
                self.search_Tree.selection_state = False
            except tk.TclError:
                pass
        else:
            video, identity, start, end, ind, state = self.model.query_label(search_state=self.search_entry.current(),
                                                                             select_ind=self.search_Tree.selection_id,
                                                                             selection_state=self.search_Tree.selection_state)
            try:
                self.search_Tree.delete(ind)
                self.search_Tree.selection_state = False
                self.Tree.selection_state = False
                self.Tree.delete(ind)
            except tk.TclError:
                pass

        self.entry1.delete(0, 'end')
        self.entry1.insert('end', video)
        self.entry3.delete(0, 'end')
        self.entry3.insert('end', identity)
        self.entry4.delete(0, 'end')
        self.entry4.insert('end', start)
        self.entry5.delete(0, 'end')
        self.entry5.insert('end', end)

        tip = self.model.rf.predict(self.model.Main_menu.X_unlab.loc[[self.model.index_name]])
        self.entry2.current(tip[0])

        self.Text.delete(1.0, 'end')

        tips = self.model.rf.predict_proba(self.model.Main_menu.X_unlab.loc[[self.model.index_name]])

        # 编制 dataframe 提示表格
        actions = ["移動Move                     ", "摂食Eat                      ", "飲水Drink                    ",
                   "羽繕いPreening                ", "身震いShivering               ", "頭かきHead scratch            ",
                   "尾振りTail swing              ", "巣箱に乗るGet on the nest  box  ", "巣箱を降りるGet off the nest box ",
                   "止まり木に乗る Get on the perch   ", "止まり木を降りる Get off the perch ", "静止 Stop                    ",
                   "休息 Rest                    ", "砂浴びDust bathing            ", "探査Litter exploration       ",
                   "首振りHead swing              ", "バランスTo keep balance        ", "センサつつきPeck the sensor      ",
                   "伸びStretching               ", "嘴とぎBeak sharpening         ", "地面つつきPeck the ground       ",
                   "きょろきょろLook around          ", "つつき攻撃Attack another hens  ", "巣箱つつきPeck the nest box     ",
                   "つつかれPecked                 ", "センサつつかれPecked the sensor   ", 'unknow']
        tips = pd.DataFrame(data={"probability": tips[0]}, index=self.model.rf.classes_)
        tips = tips.sort_values(by='probability', ascending=False)
        pd.set_option('display.unicode.ambiguous_as_wide', True)
        pd.set_option('display.unicode.east_asian_width', True)

        percentage = self.model.Main_menu.X_lab.shape[0] / 3400
        formatted_percentage = f"{percentage:.2%}"

        l = pd.DataFrame([self.model.Main_menu.X_lab.shape[0], self.model.Main_menu.X_unlab.shape[0],
                          formatted_percentage], index=['lab pool:', 'unlab pool:', 'progress:'])

        self.Text.insert('end', str(tabulate(tips, headers='keys', tablefmt='orgtbl')))
        self.Text.insert('end', '\n')
        self.Text.insert('end', tabulate(l, tablefmt='simple'))

        print("是生死时速是", self.model.Main_menu.X_unlab)

        self.model.Main_menu.X_unlab.drop(axis=0, index=self.model.index_name, inplace=True)
        # self.Main_menu.y_unlab.drop(axis=0, index=self.index_name, inplace=True)
        self.model.Main_menu.start_name = self.model.Main_menu.start_name.drop(axis=0, index=self.model.index_name)
        self.model.Main_menu.end_name = self.model.Main_menu.end_name.drop(axis=0, index=self.model.index_name)
        self.model.Main_menu.video_name = self.model.Main_menu.video_name.drop(axis=0, index=self.model.index_name)
        self.model.Main_menu.identity = self.model.Main_menu.identity.drop(axis=0, index=self.model.index_name)
        print("是生死时速是", self.model.Main_menu.X_unlab)

        # else:
        #     video, identity, start, end, ind, state = self.model.query_label(search_state=self.search_entry.current(),
        #                                                                      select_ind=self.no_label_Tree.selection_id,
        #                                                                      selection_state=self.no_label_Tree.selection_state)
        #     self.entry1.delete(0, 'end')
        #     self.entry1.insert('end', video)
        #     self.entry3.delete(0, 'end')
        #     self.entry3.insert('end', identity)
        #     self.entry4.delete(0, 'end')
        #     self.entry4.insert('end', start)
        #     self.entry5.delete(0, 'end')
        #     self.entry5.insert('end', end)
        #     try:
        #         self.no_label_Tree.selection_state = False
        #         self.no_label_Tree.delete(ind)
        #
        #     except tk.TclError:
        #         pass

    def parallel_train(self):
        # thread1 = threading.Thread(target=self.model.model_update )
        # thread1.start()
        self.model.model_update()

    def train(self):
        # if self.model.Main_menu.label_exist.get():

        lab, tree_del = self.model.trainingXXX(y=self.entry2.get_value())
        if tree_del == True:
            print("\n\n\n\n\n", "进入刷新模式了啊啊\n\n\n\n\n\n")
            self.parallel_train()
            self.model.delete_tree = False
        else:
            if lab == 'exist':
                self.Text.delete('1.0', 'end')
                self.Text.insert('end', 'wrong !!!\n' + str(self.model.index_name) + " already exists")
            else:
                self.Text.delete('1.0', 'end')
                self.Text.insert('end', 'label success' + str(lab) + "\n")
        # else:
        #     self.model.trainingXXX(y=self.entry2.get_value())

    def search(self):
        ind = self.model.search_(self.search_entry.current())
        if not ind.any():
            tk.messagebox.showwarning('No find', message="Currently No Unlab!!!")
        else:
            print(ind)

            if self.search_Tree.scrollbar:
                for h in self.search_Tree.get_children():
                    self.search_Tree.delete(h)
                self.search_Tree.scrollbar.destroy()
                self.search_Tree.set_(self.model.tree_form(inds_=self.model.search_index))
            else:
                tree = self.model.tree_form(inds_=ind)
                self.Text.delete('1.0', 'end')
                self.Text.insert('end', 'possible index:' + "\n" + str(ind))
                self.search_Tree.set_(tree)

            self.model.delete_tree = False
            self.frame_all['Search_Tree'].tkraise()

    def cancel_step(self):
        print("aaaaaaaaaaa")
        a = self.model.cancel()
        self.Text.insert('end',
                         "\n\ndelete successufl, current lab size: " + str(self.model.Main_menu.X_lab.shape[0]) + "\n")

    def skip(self):

        self.videos = self.model.Main_menu.video_path
        for i in self.videos:
            if str(self.entry1.get()) in str(i):
                self.fp = i
                self.movie = cv2.VideoCapture(i)
        self.input = av.open(self.fp)
        self.stream = self.input.streams.video[0]

        def get_point(video):
            import datetime
            a, b, c, d = "0403.wmv", "0608.wmv", '0713.wmv', "0720.wmv"
            v_dict = {a: '11:23:49.986', b: '10:18:19.591',
                      c: '12:12:45.550', d: '13:05:56.705'}

            #
            #
            # if video in a:
            #     video = a
            # if video in b:
            #     video = b
            # if video in c:
            #     video = c
            # if video in d:
            #     video = d
            #
            for i in self.model.Main_menu.V_dict.keys():
                if video in i:
                    video = i
            v_dict = self.model.Main_menu.V_dict
            time = datetime.datetime.strptime(v_dict[video], "%H:%M:%S.%f")
            deltatime = datetime.timedelta(hours=time.hour, minutes=time.minute, seconds=time.second,
                                           microseconds=time.microsecond)
            return deltatime

        start = datetime.strptime(self.entry4.get(), '%H:%M:%S.%f').time()
        self.times = start.hour * 60 * 60 + start.second + start.minute * 60 + (start.microsecond / 1000000)
        self.times = self.times - get_point(self.entry1.get()).total_seconds()
        self.times = round(self.times, 3)
        end = datetime.strptime(self.entry5.get(), '%H:%M:%S.%f').time()
        self.end = end.hour * 60 * 60 + end.second + end.minute * 60
        self.end = self.end - get_point(self.entry1.get()).total_seconds()

        self.Tscale.configure(to=int(self.stream.duration / 1000), from_=0)
        self.input.seek(offset=int(self.times * 1000), unsupported_frame_offset=False, backward=True,
                        stream=self.stream, any_frame=True)

        for i in self.input.decode(self.stream):
            self.orient = i.pts
            if np.abs(int(self.times * 1000) - i.pts) < 64:
                break
        if (self.orient < int((self.end) * 1000 + 640)):
            self.play(self.end)
            self.flag = 0
            # print("外面是:",self.orient, int((self.end) * 1000))
        self.Video_Canvas.create_image(0, 0, anchor='nw', image=self.img2)
        self.Video_Canvas.update()

    def judge(self):
        if self.flag == 1:
            self.flag = 0
            # print("进入暂停状态")
            self.pause()


        elif self.flag == 0:
            self.flag = 1
            # print("进入播放状态")
            # only flag=0 ,play the video, otherwise, yellow label.
            while self.flag == 1:
                self.play()

    def pause(self):
        self.Video_Canvas.create_image(0, 0, anchor='nw', image=self.img2)

    def play(self, end=None):
        if self.model.Main_menu.time_speed:
            self.time_speed = self.model.Main_menu.speed_get()
        if end:
            count = 0
            a = time.time()
            for i in self.input.decode(self.stream):
                self.orient = score = i.pts
                i.reformat(width=640, height=480)
                i = i.to_image()
                self.img2 = ImageTk.PhotoImage(i)
                self.Video_Canvas.create_image(0, 0, anchor='nw', image=self.img2)
                self.Tscale.set(int(score / 1000))
                self.Video_Canvas.update()
                # print(self.time_speed)
                time.sleep(self.time_speed)
                count += 1

                if self.orient > int((self.end) * 1000 + 736):
                    b = time.time()
                    # print("time_speed:, 0.5倍", (((count / 16 )/ (b - a)) - 1) * ((b - a) / count))
                    # print("time_speed:, 0.75倍",(((count / 12 )/ (b - a)) - 1) * ((b - a) / count))
                    # print("time_speed:, 1s", (((count / 32 )/ (b - a)) - 1) * ((b - a) / count))
                    # print("time_speed:, 1.25倍", (((count / 40 )/ (b - a)) - 1) * ((b - a) / count))
                    # print("time_speed:, 1.5倍", (((count / 48 )/ (b - a)) - 1) * ((b - a) / count))
                    # print("time_speed:, 2倍", (((count / 64) / (b - a)) - 1) * ((b - a) / count))
                    # print("time_speed:, 2倍", (count / (32 * 2) / (b-a) - 1) * ((b-a) / count))
                    self.flag = 0
                    break
        else:
            for i in self.input.decode(self.stream):
                if self.flag == 0:
                    break
                self.orient = score = i.pts
                i.reformat(width=640, height=480)
                i = i.to_image()
                img2 = ImageTk.PhotoImage(i)
                self.img2 = img2
                self.Video_Canvas.create_image(0, 0, anchor='nw', image=img2)
                self.Tscale.set(int(score / 1000))
                self.Video_Canvas.update()
                time.sleep(self.time_speed)

    def rais(self):
        import pandas as pd
        # if self.model.Main_menu.label_exist.get() == True:
        print(len(self.search_dic))
        if self.search_entry.current() != -1 and self.search_entry.current() != len(self.search_dic) - 1:
            if self.present == 0:
                self.frame_all['Search_Tree'].tkraise()
                self.present = 1
            else:
                self.frame_all['Text'].tkraise()
                self.present = 0
        else:
            if self.present == 0:
                self.frame_all['Tree'].tkraise()
                self.present = 1
            else:
                self.frame_all['Text'].tkraise()
                self.present = 0

        # else:
        #     self.frame_all['No label'].tkraise()
        #     self.present = 0
