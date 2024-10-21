import tkinter as tk
from tkinter import messagebox
import requests


class LoginPage(tk.Frame):
    def __init__(self, parent, controller):
        super().__init__(parent)
        self.controller = controller

        self.label_username = tk.Label(self, text="Email")
        self.label_username.place(relx=0.5, rely=0.4, anchor='center')

        self.entry_username = tk.Entry(self, show="")
        self.entry_username.place(relx=0.5, rely=0.45, anchor='center')

        self.label_password = tk.Label(self, text="Password")
        self.label_password.place(relx=0.5, rely=0.5, anchor='center')

        self.entry_password = tk.Entry(self, show="*")
        self.entry_password.place(relx=0.5, rely=0.55, anchor='center')

        self.button_login = tk.Button(self, text="Login", command=self.login)
        self.button_login.place(relx=0.5, rely=0.6, anchor='center')


    def login(self):
        email = self.entry_username.get()
        password = self.entry_password.get()

        base_url = 'https://ivaktvision-api-dev-4776e754665f.herokuapp.com/'
        auth_endpoint = ('accounts/login/')
        

        r = requests.post(base_url + auth_endpoint,
                          data={'email': email, 'password': password})

        if r.status_code == 200:
            data = r.json()
            self.controller.refresh_token = data['token']['refresh']
            self.controller.access_token = data['token']['access']
            self.get_shopdata()
            self.controller.show_frame("SelectWorkspace")
        else:
            messagebox.showerror("Login Failed", "Invalid Username or Password")

    def update_data(self):
        return None
    
    def get_shopdata(self):
        base_url = 'https://ivaktvision-api-dev-4776e754665f.herokuapp.com/'
        shops_endpoint = ('shops/')
        r = requests.get(base_url + shops_endpoint,
                        headers={"Authorization":
                                f"Bearer {self.controller.access_token}"})
        if r.status_code == 200:
            print(r.json())
            active_shops = [shop for shop in r.json() if shop['is_active'] == True]
            for shop in active_shops:
                self.controller.shop_data[shop['uuid']] = shop['shop_name']
        else:
            print(r.status_code)
