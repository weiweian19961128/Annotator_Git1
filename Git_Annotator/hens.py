from HENS.application import Application

from tkinter.ttk import Style

app = Application()
style =  Style(app)
# ('winnative', 'clam', 'alt', 'default', 'classic', 'vista', 'xpnative')
style.theme_use('winnative')

# style.theme_use('clam')
import ctypes
ctypes.windll.shcore.SetProcessDpiAwareness(1)
app.mainloop()


