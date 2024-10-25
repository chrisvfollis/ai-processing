import tkinter as tk
from tkinter import Canvas
from PIL import Image, ImageTk
import requests
import csv
import os
import sqlite3


class SelectWorkspace(tk.Frame):
    def __init__(self, parent, controller):
        super().__init__(parent)

        self.controller = controller

        self.selected_shop = tk.StringVar(self)
        self.selected_shop.set("Select a workspace")

        self.input_label = tk.Label(self, text="Select Workspace")
        self.input_label.place(relx=0.5, rely=0.4, anchor='center')

        options = list(self.controller.shop_data.values()) if self.controller.shop_data else ["No shops available"]
        self.dropdown = tk.OptionMenu(self, self.selected_shop, *options)
        self.dropdown.place(relx=0.5, rely=0.5, anchor='center')

        self.submit_button = tk.Button(self, text="Submit", command=self.submit)
        self.submit_button.place(relx=0.5, rely=0.6, anchor='center')

    def update_data(self):
        options = list(self.controller.shop_data.values()) if self.controller.shop_data else ["No shops available"]

        menu = self.dropdown["menu"]
        menu.delete(0, "end")

        for option in options:
            menu.add_command(label=option, command=lambda value=option: self.selected_shop.set(value))

        if options:
            self.selected_shop.set(options[0])
    
    def submit(self):
        def _save_shop(db_path):
            conn = sqlite3.connect(db_path)
            cursor = conn.cursor()

            cursor.execute('''
                CREATE TABLE IF NOT EXISTS shop (
                    uuid TEXT PRIMARY KEY,
                    shop_name TEXT NOT NULL
                )
            ''')
            cursor.execute('DELETE FROM shop')
            conn.commit()

            selected_workspace = self.selected_shop.get()
            self.controller.selected_shop = selected_workspace

            shop_uuid = None
            for uuid, shop_name in self.controller.shop_data.items():
                print(uuid)
                print(shop_name)
                if shop_name == selected_workspace:
                    shop_uuid = uuid
                    break
            if shop_uuid:
                cursor.execute('''
                    INSERT OR REPLACE INTO shop (
                        uuid, shop_name
                    ) VALUES (?, ?)
                ''', (shop_uuid, shop_name))

                conn.commit()
            else:
                print('No shop UUID')

            conn.close()
            return shop_uuid
        
        def _fetch_employee_data(shop_uuid):
            base_url = 'https://timemanager-api-dev-b944386035a1.herokuapp.com/'
            employees_endpoint = 'employees-json/'
            url = f"{base_url}{employees_endpoint}?shop_uuid={shop_uuid}"
            headers = {
                "X-Custom-API-Key": self.controller.api_key,
                "Authorization": f"Bearer {self.controller.access_token}"
            }
            response = requests.get(url, headers=headers)
            if response.status_code == 200:
                employee_data = response.json().get('employees', [])
                for employee in employee_data:
                    print(employee['first_name'])
                    link = employee['front_image']
                    r = requests.get(link)
                    if response.status_code == 200:
                        filename = link.rsplit('/', 1)[1]
                        filename = filename.rsplit('_', 1)[0] + '.' + filename.rsplit('_', 1)[1]
                        with open(f'../input_files/faces/{filename}', 'wb') as file:
                            file.write(r.content)
                        employee['front_image'] = filename
                    else:
                        print(response.status_code)
                        print(response.text)
            else:
                print(response.status_code)
                print(response.text)
            
            return employee_data

        def _save_employee_data(db_path, employee_data):
            """Inserts employee data into the employees table."""
            conn = sqlite3.connect(db_path)
            cursor = conn.cursor()

            cursor.execute('''
                CREATE TABLE IF NOT EXISTS employees (
                    uuid TEXT PRIMARY KEY,
                    first_name TEXT NOT NULL,
                    last_name TEXT NOT NULL,
                    shop_uuid TEXT NOT NULL,
                    front_image TEXT
                )
            ''')
            cursor.execute('DELETE FROM employees')
            conn.commit()

            insert_query = '''
                INSERT OR REPLACE INTO employees (
                    uuid, first_name, last_name, shop_uuid, front_image
                ) VALUES (?, ?, ?, ?, ?)
            '''

            for employee in employee_data:
                cursor.execute(insert_query, (
                    employee['uuid'],
                    employee['first_name'],
                    employee['last_name'],
                    employee['shop_uuid'],
                    employee['front_image']
                ))

            conn.commit()
            conn.close()
    
        shop_uuid = _save_shop('../appdata/data.db')
        if shop_uuid:
            employee_data = _fetch_employee_data(shop_uuid)
            _save_employee_data('../appdata/data.db', employee_data)
        else:
            print('no uuid')
        self.controller.show_frame('AnnotateImages')


class AnnotateImages(tk.Frame):
    def __init__(self, parent, controller, image_dir="../appdata/images"):
        super().__init__(parent)
        self.controller = controller

        self.canvas = Canvas(self)
        self.canvas.grid(row=0, column=0, sticky="nsew")
        self.bind("<Configure>", self.on_resize)
        
        self.image_dir = image_dir
        self.image_files = sorted([f for f in os.listdir(self.image_dir) if
                                   f.lower().endswith(('.png', '.jpg',
                                                       '.jpeg'))])
        self.image_index = 0
        self.original_img = None
        self.original_w, self.original_h = None, None
        self.load_image(event=True)

        self.points = []
        self.display_points = []
        self.lines = []
        self.box_counter = 0

        self.offset_x = 0
        self.offset_y = 0

        self.canvas.bind("<Button-1>", self.on_click)
        self.canvas.bind_all('<Command-z>', self.undo_last_point)
        self.bind_all('<Control-z>', self.undo_last_point)

        # BUTTONS:
        button_frame = tk.Frame(self)
        button_frame.grid(row=1, column=0, pady=(10, 20))

        self.prev_button = tk.Button(button_frame, text="< Previous",
                                     command=self.previous_image)
        self.prev_button.grid(row=0, column=0, padx=10)

        self.save_button=tk.Button(button_frame, text="Save Annotations",
                                   command=self.save_annotations)
        self.save_button.grid(row=0, column=1, padx=10)

        self.skip_button = tk.Button(button_frame, text="Next >",
                                     command=self.next_image)
        self.skip_button.grid(row=0, column=2, padx=10)

        self.grid_rowconfigure(0, weight=1)
        self.grid_columnconfigure(0, weight=1)

    def load_image(self, event=False):
        self.reset_annotations()
        image_path = os.path.join(self.image_dir, self.image_files[self.image_index])
        self.original_img = Image.open(image_path)
        self.original_w, self.original_h = self.original_img.size
        if not event:
            self.on_resize(None)
    
    def previous_image(self):
        if self.image_index > 0:
            self.image_index -= 1
            self.load_image()

    def next_image(self):
        if self.image_index < len(self.image_files) - 1:
            self.image_index += 1
            self.load_image()
        elif self.image_index == len(self.image_files) - 1:
            self.controller.show_frame('SelectPrimary')

    def on_resize(self, event):
        if event:
            self.canvas_w = event.width
            self.canvas_h = event.height
        
        self.reset_annotations()

        self.display_img = self.original_img.copy()
        self.display_img.thumbnail((self.canvas_w, self.canvas_h),
                                   Image.Resampling.LANCZOS)
        self.display_w, self.display_h = self.display_img.size

        self.offset_x = (self.canvas_w - self.display_w) // 2
        self.offset_y = (self.canvas_h - self.display_h) // 2

        self.canvas.config(width=self.canvas_w, height=self.canvas_h)
        self.photo = ImageTk.PhotoImage(self.display_img)
        self.canvas.create_image(self.offset_x, self.offset_y, anchor="nw", image=self.photo)

    def on_click(self, event):
        if (self.offset_x <= event.x <= self.offset_x + self.display_w and
            self.offset_y <= event.y <= self.offset_y + self.display_h):
            
            x_on_image = event.x - self.offset_x
            y_on_image = event.y - self.offset_y
            
            x_scaled = int(x_on_image * (self.original_w / self.display_w))
            y_scaled = int(y_on_image * (self.original_h / self.display_h))

            self.display_points.append((x_on_image, y_on_image))
            self.points.append((x_scaled, y_scaled))

            self.canvas.create_oval(event.x - 3, event.y - 3, event.x + 3, event.y + 3, fill='red')

            point_index = len(self.display_points)
            box_position = (point_index - 1) % 4
            if box_position == 3:
                self.lines.append(self.canvas.create_line(
                    self.display_points[-2][0] + self.offset_x, self.display_points[-2][1] + self.offset_y,
                    event.x, event.y, fill='red'))
                self.lines.append(self.canvas.create_line(
                    self.display_points[-4][0] + self.offset_x, self.display_points[-4][1] + self.offset_y,
                    event.x, event.y, fill='red'))
            elif box_position == 0:
                pass
            else:
                last_point_display = self.display_points[-2]
                self.lines.append(self.canvas.create_line(
                    last_point_display[0] + self.offset_x, last_point_display[1] + self.offset_y,
                    event.x, event.y, fill='red'))

    def undo_last_point(self, event=None):
        if len(self.display_points) == 0:
            return None
        
        self.display_points.pop()
        self.points.pop()
        if len(self.lines) > 0:
            line_id = self.lines.pop()
            self.canvas.delete(line_id)
        
        self.redraw_annotations()

    def reset_annotations(self):
        self.points = []
        self.display_points = []
        self.lines = []
        self.box_counter = 0
        self.canvas.delete('annotation')

    def redraw_annotations(self, saved=False):
        if saved:
            color = 'green'
        else:
            color = 'red'

        self.canvas.delete("all")
        self.photo = ImageTk.PhotoImage(self.display_img)
        
        
        self.canvas.create_image(self.offset_x, self.offset_y, anchor="nw", image=self.photo)

        offset_points = [(x + self.offset_x, y + self.offset_y) for (x, y) in self.display_points]
        for point in offset_points:
            self.canvas.create_oval(point[0] - 3, point[1] - 3,
                                    point[0] + 3, point[1] + 3, fill=color)
        
        for i in range(0, len(self.display_points)):
            box_position = i % 4
            if box_position == 3:
                self.canvas.create_line(offset_points[i-1][0],
                                        offset_points[i-1][1],
                                        offset_points[i][0],
                                        offset_points[i][1], fill=color)
                self.canvas.create_line(offset_points[i-3][0],
                                        offset_points[i-3][1],
                                        offset_points[i][0],
                                        offset_points[i][1], fill=color)
            elif box_position != 0:
                self.canvas.create_line(offset_points[i-1][0],
                                        offset_points[i-1][1],
                                        offset_points[i][0],
                                        offset_points[i][1], fill=color)

    def save_annotations(self):
        cam = self.image_files[self.image_index].split('.')[0]
        with open(f'../config/{cam}_entryways.csv', 'w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(['Entrance', 'X', 'Y'])

            for i, point in enumerate(self.points):
                box_number = (i // 4)
                writer.writerow([box_number, point[0], point[1]])

        self.redraw_annotations(saved=True)
    
    def update_data(self):
        return None


class SelectPrimary(tk.Frame):
    def __init__(self, parent, controller, image_dir="../appdata/images"):
        super().__init__(parent)
        self.controller = controller

        self.is_checked = tk.BooleanVar()

        self.canvas = Canvas(self)
        self.canvas.grid(row=0, column=0, sticky="nsew")
        self.bind("<Configure>", self.on_resize)
        
        self.image_dir = image_dir
        self.image_files = sorted([f for f in os.listdir(self.image_dir) if
                                   f.lower().endswith(('.png', '.jpg',
                                                       '.jpeg'))])
        self.designations = [-1 for _ in range(len(self.image_files))]
        self.image_index = 0
        self.original_img = None
        self.original_w, self.original_h = None, None
        self.load_image(event=True)

        self.offset_x = 0
        self.offset_y = 0

        # BUTTONS:
        button_frame = tk.Frame(self)
        button_frame.grid(row=1, column=0, pady=(10, 20))

        self.prev_button = tk.Button(button_frame, text="< Previous",
                                     command=self.previous_image)
        self.prev_button.grid(row=0, column=0, padx=10)

        self.save_button=tk.Checkbutton(button_frame, text="Primary View",
                                        variable=self.is_checked,
                                        command=self.on_check)
        self.save_button.grid(row=0, column=1, padx=10)

        self.skip_button = tk.Button(button_frame, text="Next >",
                                     command=self.next_image)
        self.skip_button.grid(row=0, column=2, padx=10)

        self.grid_rowconfigure(0, weight=1)
        self.grid_columnconfigure(0, weight=1)

    def on_check(self):
        self.designations[self.image_index] *= -1
    
    def update_checkbutton_state(self):
        if self.designations[self.image_index] == 1:
            self.is_checked.set(True)
        else:
            self.is_checked.set(False)

    def load_image(self, event=False):
        image_path = os.path.join(self.image_dir, self.image_files[self.image_index])
        self.original_img = Image.open(image_path)
        self.original_w, self.original_h = self.original_img.size
        if not event:
            self.on_resize(None)
        self.update_checkbutton_state()
    
    def previous_image(self):
        if self.image_index > 0:
            self.image_index -= 1
            self.load_image()

    def next_image(self):
        if self.image_index < len(self.image_files) - 1:
            self.image_index += 1
            self.load_image()
        elif self.image_index == len(self.image_files) - 1:
            self.save_designations()

    def on_resize(self, event):
        if event:
            self.canvas_w = event.width
            self.canvas_h = event.height

        self.display_img = self.original_img.copy()
        self.display_img.thumbnail((self.canvas_w, self.canvas_h),
                                   Image.Resampling.LANCZOS)
        self.display_w, self.display_h = self.display_img.size

        self.offset_x = (self.canvas_w - self.display_w) // 2
        self.offset_y = (self.canvas_h - self.display_h) // 2

        self.canvas.config(width=self.canvas_w, height=self.canvas_h)
        self.photo = ImageTk.PhotoImage(self.display_img)
        self.canvas.create_image(self.offset_x, self.offset_y, anchor="nw", image=self.photo)

    def save_designations(self):
        conn = sqlite3.connect('../appdata/data.db')
        cursor = conn.cursor()
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS camera_designations (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                camera TEXT NOT NULL,
                designation TEXT NOT NULL
            )
        ''')

        cursor.execute('DELETE FROM camera_designations')
        conn.commit()


        for file in self.image_files:
            i = self.image_files.index(file)
            designation = self.designations[i]
            cam = file.split('.')[0]

            if designation == -1:
                designation_str = 'secondary'
            elif designation == 1:
                designation_str = 'primary'
            
            cursor.execute('''
                INSERT INTO camera_designations (camera, designation)
                VALUES (?, ?)
            ''', (cam, designation_str))

        conn.commit()
        conn.close()

    def update_data(self):
        return None