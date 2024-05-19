import pandas as pd
import numpy as np
from scipy import signal
from scipy import stats
from statistics import mean, variance, stdev
from tkinter import ttk
import tkinter as tk
from tkinter import filedialog
# Task, raw data(aXYZ, gXYZ) ---> RF feature data
    #  1. raw into segmentation
    #  2. transform with numpy

names = ['ave_x', 'ave_y', 'ave_z',
         'ave_gx', 'ave_gy', 'ave_gz', 'ave_m',
         'var_x', 'var_y', 'var_z',
         'var_gx', 'var_gy', 'var_gz', 'var_m',
         'std_x', 'std_y', 'std_z',
         'std_gx', 'std_gy', 'std_gz', 'std_m',
         'iqr_x', 'iqr_y', 'iqr_z',
         'iqr_gx', 'iqr_gy', 'iqr_gz', 'iqr_m',
         'mad_x', 'mad_y', 'mad_z',
         'mad_gx', 'mad_gy', 'mad_gz', 'mad_m',
         'mead_x', 'mead_y', 'mead_z',
         'mead_gx', 'mead_gy', 'mead_gz', 'mead_m',
         'skew_x', 'skew_y', 'skew_z',
         'skew_gx', 'skew_gy', 'skew_gz', 'skew_m',
         'kurt_x', 'kurt_y', 'kurt_z',
         'kurt_gx', 'kurt_gy', 'kurt_gz', 'kurt_m',
         'mc_x', 'mc_y', 'mc_z',
         'mc_gx', 'mc_gy', 'mc_gz', 'mc_m',
         'spectol_x', 'spectol_y', 'spectol_z',
         'spectol_gx', 'spectol_gy', 'spectol_gz','spectol_m' ,
         'energy_x', 'energy_y', 'energy_z',
         'energy_gx', 'energy_gy', 'energy_gz', 'energy_m',
         'entropy_x', 'entropy_y', 'entropy_z',
         'entropy_gx', 'entropy_gy', 'entropy_gz', 'entropy_m',
         'min_x', 'min_y', 'min_z',
         'min_gx', 'min_gy', 'min_gz', 'min_m',
         'max_x', 'max_y', 'max_z',
         'max_gx', 'max_gy', 'max_gz', 'max_m',
         'corr_xy', 'corr_yz', 'corr_zx', 'p_xy', 'p_yz', 'p_zx', 'start','end']

start_point = None


def mac(sequces):
    ave = np.mean(sequces)
    cross = np.sign(sequces - ave)
    count = 0
    for i in range(cross.shape[0]-1):
        if cross[i] + cross[i+1] == 0:
            count+=1
    return count

def mean_cross(data):
    ave = np.mean(data)
    count = 0
    pin = None
    if data[0] < ave:
        pin = 0
    else:
        pin = 1
    for i in data:
        if pin == 0:
            if i >= ave:
                count += 1
                pin = 1
        else:
            if i < ave:
                count += 1
                pin = 0
    return count

def spectol(data):
    half = int(data.shape[0]/2)
    amp = np.abs(data)
    return amp[0:half].argmax()





class app2(tk.Frame):
    def __init__(self, master=None):
        super().__init__(master)
        self.data = None
        self.progress = tk.IntVar()
        self.text_progress = tk.StringVar()
        self.op_btn = ttk.Button(master, text='open',width=10,command= lambda: self.data_add(master))
        self.save_btn = ttk.Button(master, text='save', width=10,command = lambda: self.data_save())
        self.transform = ttk.Button(master, text= 'transform', width=10, command= lambda: self.data_trans())
        self.master = master
        self.pb = None



        self.op_btn.grid(row=0, column=0, padx=50,pady=80)
        self.transform.grid(row=0, column=1)
        self.save_btn.grid(row=0, column=2, padx=50,pady=1)


        self.entry_video = ttk.Entry(master)
        self.entry_target = ttk.Entry(master)
        self.entry_color = ttk.Entry(master)

        self.labe_video = ttk.Label(master, text='video')
        self.labe_target = ttk.Label(master, text='target')
        self.labe_color = ttk.Label(master, text='color')

        self.labe_video.grid(row=1, column=2, sticky='n')
        self.entry_video.grid(row=2, column=2, sticky='n')
        self.labe_target.grid(row=3, column=2)
        self.entry_target.grid(row=4, column=2)
        self.labe_color.grid(row=5, column=2)
        self.entry_color.grid(row=6, column=2)
    def data_add(self, master):
        path = filedialog.askopenfilenames(title='select raw data file')
        print(path)
        self.data =  pd.concat([pd.read_csv(i, encoding='shift_jis') for i in path], axis= 0)
        print(self.data)
        #self.data =pd.DataFrame(data=self.data.values)


        columns_in=self.data.columns
        canvas_x = tk.Canvas(master, bg='white', width=120, height=130)
        canvas_x.pack_propagate(False)
        canvas_x.grid(row=1, column = 0, rowspan=6)

        def gen_columns(columns_in, master):
            value_list, button_list = [], []
            for count, i in enumerate(columns_in):
                value = tk.IntVar()
                c = tk.Checkbutton(master, variable=value, text=i, bg='white')
                master.create_window(0, 22 * count, anchor='w', window=c)
                value_list.append(value)
                button_list.append(c)
            return value_list, button_list
        def gen_scrollbar(master):
            # 滚动条必须后生成 否则会出错
            def _scroll(event, widget):
                widget.yview_scroll(int(-1 * (event.delta / 120)), "units")
            scroll_x = tk.Scrollbar(master, orient='vertical', width=14)
            scroll_x.pack(side='right', fill='y')
            master.config(yscrollcommand=scroll_x.set, scrollregion=master.bbox('all'))
            master.bind("<MouseWheel>", lambda event, widget=master: _scroll(event, master))
            scroll_x.config(command=master.yview)

        self.list1, self.list1_bt = gen_columns(columns_in, canvas_x)
        gen_scrollbar(canvas_x)

    def feature_extraction(self,data, self_data ):
        # segmentation
        final = []
        list = []
        data['m'] = np.sqrt(np.square(data.iloc[:, 0]) + np.square(data.iloc[:, 1]) + np.square(data.iloc[:, 2]))
        print(data['m'])
        time = []
        end_time = []

        # for i in range(122, 12200 ,51):
        # 64/50% overlap， 128 window size

        for i in range(0, data.shape[0], 64):
            if (i + 128) < data.shape[0]:
                list.append(data.iloc[i:i + 128])
                time.append(self_data.iloc[i, 1])
                end_time.append(self_data.iloc[i + 128, 1])

        self.pb["maximum"] = len(list)
        self.pb["value"] = 0

        for count, sample in enumerate(list):
            print(count/len(list),self.progress.get())
            self.progress.set(count)
            self.text_progress.set("{:.3}".format(count/len(list)*100)+'%')

            unit = []
            for column in sample.columns:
                feature = sample[column].values
                ave = np.mean(feature)
                var = variance(feature)
                std = stdev(feature)
                iqr = stats.iqr(feature)
                mad = sample[column].mad()
                mead = (sample[column] - sample[column].median()).abs().median()
                skew = stats.skew(feature)
                kurt = stats.kurtosis(feature)
                max = np.max(feature)
                min = np.min(feature)
                mc = mean_cross(feature)  # mac(feature)

                # f =  signal.spectrogram(feature, 1.6)
                # f = np.array([np.array(h) for i in f for h in i]).flatten()
                # print(f)

                f = np.fft.fft(feature)
                half = int(f.shape[0] / 2)

                spect = spectol(f)

                energy = np.sum((np.abs(f) ** 2)[0:half])
                # a = np.sum(np.square(f / sum(f)))
                a = np.sum(np.abs(f[0:half]) ** 2)

                # print(energy, a)
                tmp = np.abs(f) ** 2 / a
                entropy = 0
                for i in range(half):
                    entropy += tmp[i] * np.log2(tmp[i])
                entropy = -entropy

                axis = pd.DataFrame([ave, var, std, iqr, mad, mead, skew, kurt, max, min, entropy, energy, mc, spect],
                                    index=[str(column) + 'ave', str(column) + 'var', str(column) + 'std',
                                           str(column) + 'iqr', str(column) + 'mad', str(column) + 'mead',
                                           str(column) + 'skew', str(column) + 'kurt', str(column) + 'max',
                                           str(column) + 'min', str(column) + 'entropy', str(column) + 'energy',
                                           str(column) + 'mc', str(column) + 'spectol'])
                unit.append(axis)

            xy, p_xy = stats.spearmanr(sample.iloc[:, 0], sample.iloc[:, 1])
            xz, p_xz = stats.spearmanr(sample.iloc[:, 0], sample.iloc[:, 2])
            yz, p_yz = stats.spearmanr(sample.iloc[:, 1], sample.iloc[:, 2])

            odata = pd.concat(unit, axis=0).T
            odata['cor_xy'] = xy
            odata['cor_xz'] = xz
            odata['cor_yz'] = yz
            odata['p_xy'] = p_xy
            odata['p_xz'] = p_xz
            odata['p_yz'] = p_yz
            final.append(odata)

        final = pd.concat(final, axis=0, ignore_index=True)
        time = pd.Series(time)
        end_time = pd.Series(end_time)

        final['start'] = time
        final['end'] = end_time
        # final['end'] = time[1:]

        print(final)
        # transform
        return final
    def data_trans(self):
        feature = []
        window = tk.Toplevel()
        window.grid()

        window.title('extracting progress')
        window.geometry("220x80+300+250")

        self.pb = ttk.Progressbar(window, variable=self.progress, length=200, mode="determinate",
                                  orient='horizontal')
        self.pb.grid(row=1,padx = 10, pady = 10)

        tk.Label(window, textvariable =self.text_progress ).grid(row=0, sticky='N')

        for i, h in zip(self.list1, self.list1_bt):
            if i.get() == 0:
                pass
            if i.get() == 1:
                c = h.cget('text')
                feature.append(c)
        print(feature)
        print(self.data.columns)
        self.data1 = self.data.loc[:, feature]
        print(self.data1.columns)
        def bbb():
            self.final = self.feature_extraction(self.data1,self.data)

        import threading as t
        t = t.Thread(target=bbb)
        t.start()
        # window.update_idletasks()

    def data_save(self):
        file_path = filedialog.asksaveasfilename(title='save processed feature file')
        print("save file", file_path)
        if file_path is not None:
            str = file_path
            #ave, var, std, iqr, mad, mead, skew, kurt, max, min, entropy, energy, mc, spect
            self.final =  self.final.iloc[:,[0,14,28,42,56,70,84,
                                          1,15,29,43,57,71,85,
                                          2,16,30,44,58,72,86,
                                          3,17,31,45,59,73,87,
                                          4,18,32,46,60,74,88,
                                          5,19,33,47,61,75,89,
                                          6,20,34,48,62,76,90,
                                          7,21,35,49,63,77,91,
                                          12,26,40,54,68,82,96,
                                          13,27,41,55,69,83,97,
                                          11,25,39,53,67,81,95,
                                          10,24,38,52,66,80,94,
                                          9,23,37,51,65,79,93,
                                          8,22,36,50,64,78,92,
                                          98,100,99,
                                          101,103,102,104,105]]

            self.final.columns= names
            self.final ['color'] = [self.entry_color.get() for i in range(self.final.shape[0])]
            self.final['video'] = [self.entry_video.get() for i in range(self.final.shape[0])]
            self.final['target']= [self.entry_target.get() for i in range(self.final.shape[0])]


            self.final.to_csv(path_or_buf=str, index= False, encoding='shift_jis')

            print(str + 'feature.csv')

# root = tk.Tk()
# root.title("feature extraction software")
# root.geometry("500x500")
# apps = app2(master = root)
# root.mainloop()



#data = pd.read_excel('mmm.xlsx')
#sr  = signal.spectrogram(data.iloc[:100,1], 89)
# from matplotlib import  pyplot as plt
# a = np.array([h for i in sr for h in i])
# print(len(sr))
# plt.plot(a)
#
# plt.show()
#
# print(np.sum([np.square(i) for i in data.iloc[:100,1]]))
# print(np.sum([np.square(i) for i in a]))




