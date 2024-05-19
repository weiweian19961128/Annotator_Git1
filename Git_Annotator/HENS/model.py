import json
from datetime import datetime
import pandas as pd

import HENS.widget as w
import matplotlib.pyplot as plt

import numpy as np

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from lightgbm import LGBMClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC



from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler



class csv_model():
    def __init__(self):
        # self.X_lab, self.y_lab,self.X_unlab, self.y_unlab, self.start_name, self.end_name, self.video_name, self.identity, self.X_test, self.y_test = None,None,None,None,None,None,None,None,None,None
        self.Main_menu = None
        self.collection, self.index_name = None, None
        self.search_index = pd.Index(data=[])
        self.count = 0
        self.margin = None
        self.margin_exist = False
        self.update_ = False
        self.delete_tree = False
        self.rf = None
        self.rf_vice = None
        self.stop_metric = []
        self.dict = None
        self.rf_available = True
        self.query_now = None


    def menu(self):
        self.Main_menu = w.Main_Menu()
        return self.Main_menu

    def Active_Learning(self):
        # 0: Margin 1: Entropy 2: Least confidence
        print("可以确定地值是:", self.Main_menu.var_AL.get())
        model = self.rf1()
        if self.Main_menu.var_AL.get() == 0:
            margin, collection, model = self.margin_rank(model, self.Main_menu.X_lab, self.Main_menu.y_lab,
                                                         self.Main_menu.X_unlab)
            return margin, collection, model
        elif self.Main_menu.var_AL.get() == 1:
            entropy, collection, model = self.entropy_rank(model, self.Main_menu.X_lab, self.Main_menu.y_lab,
                                                           self.Main_menu.X_unlab)
            return entropy, collection, model
        elif self.Main_menu.var_AL.get() == 2:
            least_confidence, collection, model = self.least_confidence_rank(model, self.Main_menu.X_lab,
                                                                             self.Main_menu.y_lab,
                                                                             self.Main_menu.X_unlab)
            return least_confidence, collection, model

    def entropy_rank(self, model, X_lab, y_lab, X_unlab):
        pd.options.display.float_format = "{:.8f}".format
        model.fit(X_lab, y_lab)  # ['target'].to_list())
        prob = model.predict_proba(X_unlab)
        X_unlab = pd.DataFrame(data=X_unlab)
        from scipy.stats import entropy
        entropy = entropy(prob, axis=1)
        X_unlab['entropy'] = entropy
        self.entropy = X_unlab['entropy']
        self.entropy = self.entropy.sort_values(ascending=False)
        print('entropy', self.collection)
        self.entropy, self.collection = self.entropy, self.entropy.index

        self.Main_menu.X_unlab = X_unlab.iloc[:, :-1]
        self.margin_exist = True
        self.delete_tree = True
        return self.entropy, self.collection, model

    def least_confidence_rank(self, model, X_lab, y_lab, X_unlab):
        pd.options.display.float_format = "{:.8f}".format
        model.fit(X_lab, y_lab)  # ['target'].to_list())
        prob = model.predict_proba(X_unlab)
        uncertainty = 1 - prob.max(axis=1)
        X_unlab['uncertainty'] = uncertainty
        self.uncertainty = X_unlab['uncertainty']
        self.uncertainty = self.uncertainty.sort_values(ascending=False)
        self.uncertainty, self.collection = self.uncertainty, self.uncertainty.index

        self.Main_menu.X_unlab = X_unlab.iloc[:, :-1]
        self.margin_exist = True
        self.delete_tree = True
        # print("主动", collection)
        return self.uncertainty, self.collection, model

    def margin_rank(self, model, X_lab, y_lab, X_unlab):
        pd.options.display.float_format = "{:.8f}".format
        # model.fit(X_lab,y_lab)
        model.fit(X_lab, y_lab)
        prob = model.predict_proba(X_unlab)
        part = np.partition(-prob, 1, axis=1)
        part = np.sort(part, axis=1)
        margin = pd.Series(- part[:, 0] + part[:, 1], index=X_unlab.index)
        X_unlab = pd.DataFrame(data=X_unlab)
        X_unlab['margin'] = margin
        # print(margin)

        X_unlab = X_unlab.sort_values(by=['margin'], axis=0, ascending=True)

        self.margin, self.collection = X_unlab['margin'], X_unlab['margin'].index
        self.Main_menu.X_unlab = X_unlab.iloc[:, :-1]
        self.margin_exist = True
        self.delete_tree = True
        print(self.margin, self.collection)
        return self.margin, self.collection, model

    def rf1(self):
        #self.rf = RandomForestClassifier(random_state=43, n_estimators=80, max_depth=12, max_features='auto',
          #                               min_samples_leaf=1, min_samples_split=5, class_weight="balanced", n_jobs=-1)
        # self.rf_vice = RandomForestClassifier(random_state=43, n_estimators=80, max_depth=12, max_features='auto',
        #                                       min_samples_leaf=1, min_samples_split=5, class_weight="balanced",
        #                                       n_jobs=-1)



        if self.Main_menu.estimator_ == 'lgr':

            lgr = LogisticRegression()
            self.rf = make_pipeline(StandardScaler(),lgr)

        elif self.Main_menu.estimator_ == 'rfr':
            self.rf = RandomForestClassifier()

        elif self.Main_menu.estimator_ == 'lgbm':
            lgbm = LGBMClassifier(verbose=-1)
            self.rf = lgbm
        elif self.Main_menu.estimator_ == 'nb':
            std = StandardScaler()
            nb = GaussianNB()
            self.rf = make_pipeline(std, nb)


        elif self.Main_menu.estimator_ == 'dt':
            std = StandardScaler()
            dt = DecisionTreeClassifier()
            self.rf = make_pipeline(std, dt)

        return self.rf

    def model_update(self):
        self.rf.fit(self.Main_menu.X_lab, self.Main_menu.y_lab)
        print("fit 完成了啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊\n\n\n\n")

    def query_label(self, search_state, select_ind, selection_state):
        print(len(self.dict) - 1, search_state, search_state == len(self.dict) - 1, "\n", select_ind, selection_state)
        if search_state == -1 or search_state == len(self.dict):
            print(self.collection)
            if selection_state == True:
                self.index_name = int(select_ind)
            else:
                self.index_name = self.collection[0]
            print(selection_state)
            print("价值的标记是", self.index_name)
            print(self.Main_menu.X_unlab.loc[self.index_name])
            self.query_now =  self.Main_menu.X_unlab.loc[self.index_name]

            start = self.Main_menu.start_name.loc[self.index_name].values[0]
            end = self.Main_menu.end_name.loc[self.index_name].values[0]
            video = self.get_video(self.Main_menu.video_name.loc[self.index_name].values[0])
            identity = self.Main_menu.identity.loc[self.index_name].values[0]
            if self.index_name in self.search_index:
                self.search_inde_decrease()
            self.collection_decrease()

            return video, identity, start, end, self.index_name, selection_state
        else:
            print(self.search_index)
            if selection_state == True:
                self.index_name = int(select_ind)
            else:
                print(selection_state)
                self.index_name = self.search_index[0]

            print("搜索的标记是", self.index_name)

            self.query_now = self.Main_menu.X_unlab.loc[self.index_name]

            start = self.Main_menu.start_name.loc[self.index_name].values[0]
            end = self.Main_menu.end_name.loc[self.index_name].values[0]
            video = self.get_video(self.Main_menu.video_name.loc[self.index_name].values[0])
            identity = self.Main_menu.identity.loc[self.index_name].values[0]
            self.search_inde_decrease()
            if self.index_name in self.collection:
                self.collection_decrease()

            return video, identity, start, end, self.index_name, selection_state
        # else:
        #     if selection_state == True:
        #         self.index_name = int(select_ind)
        #         start = self.Main_menu.start_name.loc[self.index_name].values[0]
        #         end = self.Main_menu.end_name.loc[self.index_name].values[0]
        #         video = self.get_video(self.Main_menu.video_name.loc[self.index_name].values[0])
        #         identity = self.Main_menu.identity.loc[self.index_name].values[0]
        #         return video, identity, start, end, self.index_name, selection_state
        #     else:
        #         print(self.index_name,self.Main_menu.ind_collection[0])
        #         self.index_name  = self.Main_menu.ind_collection[0]
        #
        #         start = self.Main_menu.start_name.loc[self.index_name].values[0]
        #         end = self.Main_menu.end_name.loc[self.index_name].values[0]
        #         video = self.get_video(self.Main_menu.video_name.loc[self.index_name].values[0])
        #         identity = self.Main_menu.identity.loc[self.index_name].values[0]
        #         self.Main_menu.ind_collection  = self.Main_menu.ind_collection[1:]
        #         return video, identity, start, end, self.index_name, selection_state

    def semi_learning(self):
        self.margin, self.collection, model = self.margin_rank(self.rf1(), self.Main_menu.X_lab, self.Main_menu.y_lab,
                                                               self.Main_menu.X_unlab)
        num = 500

        e = int(self.collection.size * 0.2)
        s = e + 500
        print("auto的代价是:", s, e)

        inds = self.collection.tolist()[-s:-e]
        # print(self.Main_menu.y_lab, self.collection)
        X = self.Main_menu.X_unlab.loc[inds]

        print(self.Main_menu.y_lab)
        self.rf.fit(self.Main_menu.X_lab, self.Main_menu.y_lab)
        y = self.rf.predict(X)
        y = pd.Series(y, index=inds)
        self.Main_menu.X_lab = pd.concat([self.Main_menu.X_lab, X])
        print("sssssssssssssssssssssss",type(self.Main_menu.y_lab), (y))
        if isinstance(self.Main_menu.y_lab, pd.DataFrame):
            self.Main_menu.y_lab = pd.concat([self.Main_menu.y_lab['target'], y])
        elif isinstance(self.Main_menu.y_lab, pd.Series):
            self.Main_menu.y_lab = pd.concat([self.Main_menu.y_lab, y])

        self.Main_menu.X_unlab.drop(axis=0, index=self.collection[-s:-e], inplace=True)
        self.Main_menu.y_lab.name = 'target'

        print(self.Main_menu.y_lab)

    def auto_query(self):
        self.margin_rank(self.rf1(), self.Main_menu.X_lab, self.Main_menu.y_lab, self.Main_menu.X_unlab)
        self.index_name = self.collection[0]

        self.Main_menu.X_lab.loc[self.index_name] = self.Main_menu.X_unlab.loc[self.index_name]
        self.Main_menu.y_lab.loc[self.index_name] = self.Main_menu.y_unlab.loc[self.index_name]

        self.Main_menu.X_unlab.drop(axis=0, index=self.index_name, inplace=True)
        self.Main_menu.y_unlab.drop(axis=0, index=self.index_name, inplace=True)
        self.Main_menu.start_name = self.Main_menu.start_name.drop(axis=0, index=self.index_name)
        self.Main_menu.end_name = self.Main_menu.end_name.drop(axis=0, index=self.index_name)
        self.Main_menu.video_name = self.Main_menu.video_name.drop(axis=0, index=self.index_name)
        self.Main_menu.identity = self.Main_menu.identity.drop(axis=0, index=self.index_name)

        self.Main_menu.X_lab, self.Main_menu.y_lab = pd.DataFrame(self.Main_menu.X_lab), pd.DataFrame(
            self.Main_menu.y_lab)
        self.Main_menu.X_unlab, self.Main_menu.y_unlab = pd.DataFrame(self.Main_menu.X_unlab), pd.DataFrame(
            self.Main_menu.y_unlab)

    def collection_decrease(self):
        print(self.collection)
        self.collection = self.collection.drop(self.index_name)

        print(self.collection)

    def search_inde_decrease(self):
        print(self.search_index)
        if self.index_name in self.search_index:
            self.search_index = self.search_index.drop(self.index_name)
        print(self.search_index)

    def tree_form(self, model=None, inds_=pd.Index(data=[])):
        if model:
            pred_model = model
        else:
            pred_model = self.rf1()
            pred_model.fit(self.Main_menu.X_lab, self.Main_menu.y_lab)
        # inds_ : 相应的序列集合(有价值集合序列索引, 搜索序列索引)
        if (inds_.any() == False):
            inds_ = self.collection
        p = pred_model.predict_proba(self.Main_menu.X_unlab.loc[inds_])
        list = []
        for x in p:
            word2 = pd.Series(x,
                              index=[str(self.dict[str(i)]) + " (" + str(round(class_pro, 2)) + ")" for i, class_pro in
                                     zip(pred_model.classes_, x)]).sort_values(axis=0,
                                                                               ascending=False).iloc[:4].index
            list.append(word2)
        d = pd.DataFrame(list,
                         index=inds_)
        # if d.shape[0] > 10000:
        #     leng = int(d.shape[0]/8)
        #     return d.iloc[:leng,:]
        return d

    def no_label_tree(self):
        return pd.concat([self.Main_menu.start_name, self.Main_menu.identity, self.Main_menu.video_name], axis=1).loc[
            self.Main_menu.ind_collection]

    def get_video(self, video):
        video = str(int(video))
        if video in ['0', '3', '6', '9']:
            return '0403'
        elif video in ['1', '4', '7', '10']:
            return '0608'
        elif video in ['12', '15', '18', '21']:
            return '0713'
        elif video in ['13', '16', '19', '22']:
            return '0720'

    def trainingXXX(self, y=1):
        if self.Main_menu.X_unlab.index.isin(np.array(self.index_name).flatten()).any() == False:
            print((self.Main_menu.X_lab == self.query_now).all(axis=1).any())
            if not (self.Main_menu.X_lab == self.query_now).all(axis=1).any():

                self.Main_menu.X_lab.loc[self.index_name] = self.query_now
                print(self.Main_menu.y_lab)
                self.Main_menu.y_lab.loc[self.index_name] = y

                # self.Main_menu.X_unlab.drop()
                # self.Main_menu.start_name = self.Main_menu.start_name.drop(axis=0, index=self.index_name)
                # self.Main_menu.end_name = self.Main_menu.end_name.drop(axis=0, index=self.index_name)
                # self.Main_menu.video_name = self.Main_menu.video_name.drop(axis=0, index=self.index_name)
                # self.Main_menu.identity = self.Main_menu.identity.drop(axis=0, index=self.index_name)

                self.Main_menu.X_lab, self.Main_menu.y_lab = pd.DataFrame(self.Main_menu.X_lab), pd.DataFrame(
                    self.Main_menu.y_lab)
                self.Main_menu.X_unlab, self.Main_menu.y_unlab = pd.DataFrame(self.Main_menu.X_unlab), pd.DataFrame(
                    self.Main_menu.y_unlab)


                if not isinstance(self.Main_menu.batch_size,str):
                    self.count += 1
                    if self.Main_menu.batch_size == self.count or ( self.Main_menu.batch_size - self.count)<=0:
                        self.margin_exist = True
                        self.count = 0
                        self.delete_tree = True
                        self.rf_available = False


                    print("程序刷新倒计时:\n", self.Main_menu.batch_size - self.count, "\n")
                print("程序刷新倒计时:\n", self.Main_menu.batch_size, "\n")
                return self.Main_menu.X_lab.shape[0], self.delete_tree

            else:
                print(self.Main_menu.X_lab == self.query_now, (self.Main_menu.X_lab == self.query_now).all(axis=1))
                print("里面的是", (self.Main_menu.X_lab == self.query_now).all(axis=1).any())
                return 'exist', False
            # else:
            #     self.Main_menu.X_lab.loc[self.index_name] = self.Main_menu.X_unlab.loc[self.index_name]
            #     self.Main_menu.y_lab.loc[self.index_name] = y
            #     print(self.Main_menu.X_lab, self.Main_menu.y_lab)
        else:
            print("外面的是", self.Main_menu.X_unlab.index.isin(np.array(self.index_name).flatten()).any())

            return 'exist', False

    def search_(self, num):
        ######################
        # num -> z
        z = num
        search_word = str(z) + ' ' + str(self.dict[str(z)])
        # print(search_word)
        if np.array(self.Main_menu.X_unlab).all() == None:
            return False
        else:
            model = self.rf1()
            model.fit(self.Main_menu.X_lab, self.Main_menu.y_lab)

            pred = model.predict(self.Main_menu.X_unlab)
            prob = model.predict_proba(self.Main_menu.X_unlab)
            sort_prob = np.sort(prob, axis=1)

            margin = sort_prob[:,-1] - sort_prob[:,-2]
            margin = pd.DataFrame(margin, columns=['margin'],index=self.Main_menu.X_unlab.index)
            margin['target'] = pred
            margin = margin.sort_values(by=['margin'], ascending=True)
            print(margin)

            ids = margin[margin['target']==z].index
            self.search_index = ids.copy()









            # list : 存search index ,list2: 存tree文本格式
            # list = []
            # list2 = []
            #
            # for x in p:
            #     word = pd.Series(x, index=[i for i in model.classes_])
            #     word = word.sort_values(axis=0, ascending=False)
            #     list.append(word)
            #     # 对应的字典形式
            #     word2 = pd.Series(x, index=[self.dict[str(i)] for i in model.classes_]).sort_values(axis=0,
            #                                                                                         ascending=False)
            # self.search_index = []
            # # list:排好的单个样本可能分布 的 列表
            # for count, i in enumerate(list):
            #     # w: 取前一位的分类标记, count: 对应真实索引
            #     w = [i for i in i.iloc[:1].index]
            #     # print(w)
            #
            #     # z: 搜索的标记, final: 存在的那个点
            #     for final in w:
            #         # print(z, type(z), final, type(final), "\ni首位:", i.iloc[:1], "\ni所有位置", i, '\n')
            #         if z == final:
            #             if i.iloc[:1].values > self.Main_menu.thresh_get():
            #                 print("被选中, index是: ", count)
            #                 self.search_index.append(count)
            # # print(self.search_index)
            # self.search_index = self.Main_menu.X_unlab.iloc[self.search_index, :].index
            return self.search_index

    def cancel(self):
        print(self.Main_menu.X_lab)
        self.Main_menu.X_lab = pd.DataFrame(self.Main_menu.X_lab).iloc[:-1]
        self.Main_menu.y_lab = pd.DataFrame(self.Main_menu.y_lab).iloc[:-1]
        print(self.Main_menu.X_lab)
        return 1

    def form_data(self, entry_info=None, selection_labels=None):
        feature = self.Main_menu.X_lab.copy()
        feature['target'] = self.Main_menu.y_lab.values.flatten()
        training_data = feature  # pd.concat([pd.DataFrame(self.Main_menu.X_lab), pd.DataFrame(self.Main_menu.y_lab)], axis=1)
        time = datetime.now().strftime('%Y_%m_%d %H_%M_%S')
        time = './AL_data/' + time + '.csv'
        training_data.to_csv(path_or_buf=time)
        print(training_data[training_data['target'].isin(selection_labels[0].keys())])
        training_data[training_data['target'].isin(selection_labels[0].keys())].to_csv(
            path_or_buf=selection_labels[1] + "/train.csv", header=True, index=False)

        # 1.video path (words)
        abs = {"video_path": [i for i in self.Main_menu.video_path]}
        dict.update(abs)
        p = {i: t for key, t in self.Main_menu.V_dict.items() for i in self.Main_menu.video_path if key in i}
        print(p)
        import json
        p = json.dumps(p)
        with open("./Resume/fix_setting/video_path.json", "w", encoding='utf-8') as f:
            f.write(p)

        # 2. lab X,y (data)
        training_data.to_csv(path_or_buf="./Resume/" + "lab" + '.csv', index=False, header=True)
        a = pd.read_csv("./Resume/lab.csv")
        a.index = a.iloc[:, 0].values
        x, y = a.iloc[:, :-1], a.iloc[:, -1]
        print(x, y)

        # 3.unlab: start, end, video, identity
        l = [self.Main_menu.start_name, self.Main_menu.end_name, self.Main_menu.video_name, self.Main_menu.identity]
        regist = pd.concat(objs=l, axis=1)
        print(regist)
        regist = pd.concat([self.Main_menu.X_unlab, regist], axis=1)
        regist.to_csv(path_or_buf="./Resume/" + "res" + '.csv', index=False, header=True)
        load = pd.read_csv("./Resume/res.csv")
        load.index = load.iloc[:, 0].values
        load = load.iloc[:, 1:]
        print(load)

        # 4. collection index, search index
        pd.Series(self.collection).to_csv(path_or_buf="./Resume/" + "collection" + '.csv', header=False)
        pd.Series(self.search_index).to_csv(path_or_buf="./Resume/" + "search_index" + '.csv', header=False)
        c = pd.read_csv("./Resume/collection.csv").iloc[:, 1:]
        try:
            s = pd.read_csv("./Resume/search_index.csv").iloc[:, 1:]
        except pd.errors.EmptyDataError:
            s = pd.DataFrame([])
        print(c.values, s.values)

        # 5. entry_info
        if entry_info:
            d = {'Video': entry_info[0], 'Behavior': entry_info[1], 'Color': entry_info[2], 'Start time': entry_info[3],
                 'End time': entry_info[4]}
            entry_info.insert(0, self.index_name)
            A = pd.DataFrame(entry_info)
            print(A)
            A.to_csv("./Resume/current.csv", header=False)
            try:
                B = pd.read_csv("./Resume/current.csv")
            except pd.errors.EmptyDataError:
                B = pd.DataFrame([])

        # fix: categories_setting and video_path

        with open('./Resume/fix_setting/categories.json', 'w') as f:
            # f.write(str(list(self.combo['values'])))
            json.dump(self.dict, f)

        print("\n\n", self.Main_menu.X_test, self.Main_menu.y_test)
        test_data = pd.concat([pd.DataFrame(self.Main_menu.X_test), pd.DataFrame(self.Main_menu.y_test)], axis=1)
        test_data.to_csv(path_or_buf="./Resume/" + "test" + '.csv', index=False, header=True)

    def resume1(self, state=1, entry_current=None):
        if state == 1:
            try:
                # 2 . lab_x _y
                a = pd.read_csv("./Resume/lab.csv")
                # a.index = a.iloc[:, 0].values
                # print(a)
                self.Main_menu.X_lab, self.Main_menu.y_lab = a.iloc[:, :-1], a.iloc[:, -1]
                print(self.Main_menu.X_lab, self.Main_menu.y_lab)
                self.rf1()
                self.rf.fit(self.Main_menu.X_lab, self.Main_menu.y_lab)
                self.Main_menu.label_exist.set(True)
            except Exception as error:
                print("出了第一个错误:", type(error).__name__)

            try:
                # 2/2 test_x,y
                b = pd.read_csv("./Resume/test.csv")
                self.Main_menu.X_test, self.Main_menu.y_test = b.iloc[:, :-1], b.iloc[:, -1]
                # 3.unlab: start, end, video, identity

                load = pd.read_csv("./Resume/res.csv")
                self.Main_menu.start_name = load[['start']]
                self.Main_menu.end_name = load[['end']]
                self.Main_menu.video_name = load[['video']]
                self.Main_menu.identity = load[['color']]

                # print("aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",load.drop['color','video', 'start','end'].head(5))
                names = load.columns.values.tolist()
                for i in ['start', 'end', 'video', 'color']:
                    names.remove(i)

                # names if we do not know, make it impossible
                #     names = ['ave_x', 'ave_y', 'ave_z',
                #              'ave_gx', 'ave_gy', 'ave_gz', 'ave_m',
                #              'var_x', 'var_y', 'var_z',
                #              'var_gx', 'var_gy', 'var_gz', 'var_m',
                #              'std_x', 'std_y', 'std_z',
                #              'std_gx', 'std_gy', 'std_gz', 'std_m',
                #              'iqr_x', 'iqr_y', 'iqr_z',
                #              'iqr_gx', 'iqr_gy', 'iqr_gz', 'iqr_m',
                #              'mad_x', 'mad_y', 'mad_z',
                #              'mad_gx', 'mad_gy', 'mad_gz', 'mad_m',
                #              'mead_x', 'mead_y', 'mead_z',
                #              'mead_gx', 'mead_gy', 'mead_gz', 'mead_m',
                #              'skew_x', 'skew_y', 'skew_z',
                #              'skew_gx', 'skew_gy', 'skew_gz', 'skew_m',
                #              'kurt_x', 'kurt_y', 'kurt_z',
                #              'kurt_gx', 'kurt_gy', 'kurt_gz', 'kurt_m',
                #              'mc_x', 'mc_y', 'mc_z',
                #              'mc_gx', 'mc_gy', 'mc_gz', 'mc_m',
                #              'spectol_x', 'spectol_y', 'spectol_z',
                #              'spectol_gx', 'spectol_gy', 'spectol_gz', 'spectol_m',
                #              'energy_x', 'energy_y', 'energy_z',
                #              'energy_gx', 'energy_gy', 'energy_gz', 'energy_m',
                #              'entropy_x', 'entropy_y', 'entropy_z',
                #              'entropy_gx', 'entropy_gy', 'entropy_gz', 'entropy_m',
                #              'min_x', 'min_y', 'min_z',
                #              'min_gx', 'min_gy', 'min_gz', 'min_m',
                #              'max_x', 'max_y', 'max_z',
                #              'max_gx', 'max_gy', 'max_gz', 'max_m',
                #              'corr_xy', 'corr_yz', 'corr_zx', 'p_xy', 'p_yz', 'p_zx']

                self.Main_menu.X_unlab = load[names]
                self.Main_menu.X_unlab.dropna(inplace=True)

            except Exception as error:
                print("出了第二个错误:", type(error).__name__)

            try:
                with open('./Resume/fix_setting/categories.json', 'r') as f:
                    # print(a, type(a))
                    # w = a.strip(')').strip('(').strip("[").strip("]").split(',')
                    # print()
                    # list = []
                    # for i in w:
                    #     list.append(i.replace("'", ""))
                    # print(list, dict, "\n", list != dict)
                    self.dict = json.load(f, encoding='shift_jis')
            except Exception as error:
                print("第三个An exception occurred:", type(error).__name__)

            # 4. collection index, search index
            c = pd.read_csv("./Resume/collection.csv").iloc[:, 1:]

            try:
                s = pd.read_csv("./Resume/search_index.csv").iloc[:, 1:]
            except pd.errors.EmptyDataError:
                s = pd.DataFrame([])
                print("第四个")
            # print(c.values, s.values)
            self.collection, self.search_index = pd.Index(c.values.flatten()), pd.Index(s.values.flatten())

            # 5. entry_info
            try:
                B = pd.read_csv("./Resume/current.csv", header=None)
            except pd.errors.EmptyDataError:
                B = pd.DataFrame([])
                print("第五个")
            import math
            if B.empty == False:
                if entry_current != -1:
                    self.index_name = int(B.iloc[0, 1])
                    video = B.iloc[1, 1]
                    current = int(B.iloc[2, 1])
                    color = B.iloc[3, 1]
                    start = B.iloc[4, 1]
                    end = B.iloc[5, 1]
                    # print(self.index_name, video, current, color, start, end)
                    return video, current, color, start, end
                else:

                    return B.iloc[1, 1], B.iloc[2, 1], B.iloc[3, 1], B.iloc[4, 1], B.iloc[5, 1]

        else:
            return

    def stop_criterion(self):
        margin_model = RandomForestClassifier(random_state=43, n_estimators=100, max_depth=28, max_features='auto',
                                              min_samples_leaf=1, min_samples_split=9, class_weight="balanced")

        margin_iter = []
        # print(self.Main_menu.X_lab.shape[0])
        total = self.Main_menu.X_lab.shape[0]
        for i in range(int(total / 7), total):
            # 左边是未标注, 右边人工标注
            x = self.Main_menu.X_lab.iloc[:i]
            y = self.Main_menu.y_lab.iloc[:i]
            # print(x,y)
            margin_model.fit(x, y)
            prob = margin_model.predict_proba(self.Main_menu.X_lab.iloc[[i], :])
            part = np.partition(-prob, 1, axis=1)
            part = np.sort(part, axis=1)
            margin = - part[:, 0] + part[:, 1]
            margin_iter.append(margin[0])

        loop = len(margin_iter)

        gradient_i = np.diff(margin_iter)
        w_i = []
        for count, i in enumerate(gradient_i):
            w_i.append(np.median(gradient_i[count:(count + 100)]))

        # print("margin 列表是:",margin_iter, "长度是:",len(margin_iter))
        # print("梯度是:", gradient_i,"长度是:",len(gradient_i))
        # print("中位数是:", w_i, "长度是:",len(w_i))
        # print("阈值是:", np.diff(w_i),"长度是:",len(np.diff(w_i)))
        return 0

    def draw_PR(self, labels, score1, name1, score2, name2, figname, estimators=None):
        fig = plt.figure(figsize=(30, 14))
        xt = np.array(range(len(labels)))

        a = plt.bar(np.array(xt) - 0.2, score1, width=0.4, label=name1)

        b = plt.bar(np.array(xt) + 0.2, score2, width=0.4, label=name2)

        plt.xticks(xt, [i for i in labels], rotation=45)
        plt.title(figname)
        plt.legend(loc='lower left')
        for index, value in enumerate(score1):
            plt.text(index - 0.5, value + 0.01, str(round(value, 2)), color='blue')
        for index, value in enumerate(score2):
            plt.text(index + 0.1, value + 0.01, str(round(value, 2)), color='orange')

        # p = 'D:/Onedrive/研究/ALdata/PIC/Video/0720/12-17/'
        # plt.savefig(p + str(estimators) + ' estimators-' + figname + "size.png", dpi=300)

    def stop_criterions(self, k):
        from scipy.stats import entropy

        prob = self.rf.predict_proba(self.Main_menu.X_unlab)
        # print(prob)
        ent = [entropy(i) for i in prob]
        # print(ent)
        print(self.margin)
        Max_entropy = np.max(ent)
        Mean_entropy = np.mean(ent)
        Margin = self.margin.iloc[0]
        mean_margin = np.mean(self.margin.to_list())

        MEE = (np.sum([1 - np.max(i) for i in prob])) / self.margin.shape[0]

        from sklearn.metrics import accuracy_score, f1_score
        pred = self.rf.predict(self.Main_menu.X_unlab)

        print(self.Main_menu.X_unlab, self.Main_menu.y_unlab)
        Acc = accuracy_score(y_true=self.Main_menu.y_unlab, y_pred=pred)
        micro_F1 = f1_score(self.Main_menu.y_unlab, pred, average='micro')
        macro_F1 = f1_score(self.Main_menu.y_unlab, pred, average='macro')

        # 假设是k = 5

        # print(self.collection[:5])
        # print(self.Main_menu.X_unlab)

        var_prob = [np.max(i) for i in self.rf.predict_proba(self.Main_menu.X_unlab.iloc[:5])]
        print(var_prob)
        var_mean = np.mean(var_prob)
        Variance_5 = np.sum([np.square(i - var_mean) for i in var_prob]) / 5

        var_prob = [np.max(i) for i in self.rf.predict_proba(self.Main_menu.X_unlab.iloc[:10])]
        var_mean = np.mean(var_prob)
        Variance_10 = np.sum([np.square(i - var_mean) for i in var_prob]) / 10

        var_prob = [np.max(i) for i in self.rf.predict_proba(self.Main_menu.X_unlab.iloc[:15])]
        var_mean = np.mean(var_prob)
        Variance_15 = np.sum([np.square(i - var_mean) for i in var_prob]) / 15

        var_prob = [np.max(i) for i in self.rf.predict_proba(self.Main_menu.X_unlab.iloc[:30])]
        var_mean = np.mean(var_prob)
        Variance_30 = np.sum([np.square(i - var_mean) for i in var_prob]) / 30

        var_prob = [np.max(i) for i in self.rf.predict_proba(self.Main_menu.X_unlab.iloc[:60])]
        var_mean = np.mean(var_prob)
        Variance_60 = np.sum([np.square(i - var_mean) for i in var_prob]) / 60

        var_prob = [np.max(i) for i in self.rf.predict_proba(self.Main_menu.X_unlab.iloc[:120])]
        var_mean = np.mean(var_prob)
        Variance_120 = np.sum([np.square(i - var_mean) for i in var_prob]) / 120

        var_prob = [np.max(i) for i in self.rf.predict_proba(self.Main_menu.X_unlab.iloc[:240])]
        var_mean = np.mean(var_prob)
        Variance_240 = np.sum([np.square(i - var_mean) for i in var_prob]) / 240

        var_prob = [np.max(i) for i in self.rf.predict_proba(self.Main_menu.X_unlab.iloc[:480])]
        var_mean = np.mean(var_prob)
        Variance_480 = np.sum([np.square(i - var_mean) for i in var_prob]) / 480

        var_prob = [np.max(i) for i in self.rf.predict_proba(self.Main_menu.X_unlab.iloc[:960])]
        var_mean = np.mean(var_prob)
        Variance_960 = np.sum([np.square(i - var_mean) for i in var_prob]) / 960

        print([Max_entropy, Mean_entropy, Margin, mean_margin, MEE, Variance_5, Variance_10, Variance_15, Variance_30,
               Variance_60, Variance_120, Variance_240, Variance_480, Acc, micro_F1, macro_F1])
        self.stop_metric.append(
            [Max_entropy, Mean_entropy, Margin, mean_margin, MEE, Variance_5, Variance_10, Variance_15, Variance_30,
             Variance_60, Variance_120, Variance_240, Variance_480, Variance_960, Acc, micro_F1, macro_F1])

    def write_stoprocess(self):
        with open("output_522.txt", "w") as txt_file:
            for line in self.stop_metric:
                txt_file.write(" ".join(str(line)).replace(" ", "") + "\n")
