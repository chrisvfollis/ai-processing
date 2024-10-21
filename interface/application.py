import tkinter as tk
from login_page import LoginPage
from setup_pages import SelectWorkspace, AnnotateImages, SelectPrimary


class Application(tk.Tk):
    def __init__(self):
        super().__init__()

        self.title('Multi-Page Application')

        self.geometry(self._geometry_string())
        self.container = self._setup_container()

        self.access_token = None
        self.refresh_token = None
        self.shop_data = {}
        self.selected_shop = None

        self.frames = {}
        for F in (LoginPage, SelectWorkspace, AnnotateImages, SelectPrimary):
            page_name = F.__name__
            frame = F(parent=self.container, controller=self)
            self.frames[page_name] = frame
            frame.grid(row=0, column=0, sticky='nsew')
        
        with open('../appdata/status.txt', 'r') as file:
            status = file.read()
        status = status.split('\n')
        if status[1][-1] == '1':
            self.show_frame('AnnotateImages')
        elif status[0][-1] == '1':
            self.show_frame('SelectWorkspace')
        elif status[0][-1] == '0':
            self.show_frame('LoginPage')
    
    def show_frame(self, page_name):
        frame = self.frames[page_name]
        frame.update_data()
        frame.tkraise()
    
    def _geometry_string(self, scale=0.7):
        screen_w = self.winfo_screenwidth()
        screen_h = self.winfo_screenheight()

        window_w = int(screen_w * scale)
        window_h = int(screen_h * scale)

        window_x = (screen_w // 2) - (window_w // 2)
        window_y = (screen_h // 2) - (window_h // 2)
        return f"{window_w}x{window_h}+{window_x}+{window_y}"
    
    def _setup_container(self):
        container = tk.Frame(self)
        container.pack(side="top", fill="both", expand=True)
        container.grid_rowconfigure(0, weight=1)
        container.grid_columnconfigure(0, weight=1)
        return container

if __name__ == '__main__':
    app = Application()
    app.mainloop()

