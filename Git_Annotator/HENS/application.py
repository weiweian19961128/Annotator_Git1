import tkinter as tk

import HENS.view as v
import HENS.model as m


class Application(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("hens")




        self.geometry(f'{1281}x{640}+250+180')
        #self.resizable(0, 0)
        model = m.csv_model()
        t = model.menu()

        self.config(menu=t.m)
        self.recordform = v.Annotate_Frame(self, model)
        self.recordform.grid(row=0, sticky=(tk.W + tk.E))
