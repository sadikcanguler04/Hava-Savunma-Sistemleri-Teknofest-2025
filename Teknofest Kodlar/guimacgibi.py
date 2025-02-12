import tkinter as tk
from tkinter import ttk
import cv2
import numpy as np
from PIL import Image, ImageTk
import threading
from queue import Queue
import time

class CameraThread(threading.Thread):
    def __init__(self, camera_index, queue, panel_id, processing_type):
        super().__init__()
        self.camera_index = camera_index
        self.queue = queue
        self.panel_id = panel_id
        self.processing_type = processing_type
        self.running = True
        self.cap = None

    def run(self):
        self.cap = cv2.VideoCapture(self.camera_index, cv2.CAP_DSHOW)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        
        while self.running:
            if self.cap is None or not self.cap.isOpened():
                break
                
            ret, frame = self.cap.read()
            if ret:
                frame = cv2.resize(frame, (320, 240))
                
                if self.processing_type == 'gray':
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                    frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2RGB)
                elif self.processing_type == 'hsv':
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
                    frame = cv2.cvtColor(frame, cv2.COLOR_HSV2RGB)
                elif self.processing_type == 'noise':
                    frame = cv2.GaussianBlur(frame, (5, 5), 0)
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                else:
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                
                if self.queue.qsize() < 2:
                    self.queue.put((self.panel_id, frame))
            
            time.sleep(0.03)

    def stop(self):
        self.running = False
        if self.cap is not None:
            self.cap.release()

class iOSStyleTheme:
    def __init__(self, root):
        self.root = root
        self.configure_style()

    def configure_style(self):
        style = ttk.Style()
        
        # iOS tarzı renkler
        bg_color = '#F2F2F7'  # Açık gri arka plan
        primary_color = '#007AFF'  # iOS mavi rengi
        secondary_color = '#5856D6'  # Mor ton
        text_color = '#000000'
        
        # Global stil ayarları
        style.theme_use('clam')
        
        # Notebook (Tab) stili
        style.configure('TNotebook', 
            background=bg_color, 
            borderwidth=0, 
            relief='flat'
        )
        style.map('TNotebook.Tab', 
            background=[('selected', primary_color), ('!selected', bg_color)],
            foreground=[('selected', 'white'), ('!selected', text_color)]
        )
        
        # Button stili
        style.configure('TButton', 
            background=primary_color, 
            foreground='white', 
            font=('San Francisco', 13),
            padding=10,
            borderwidth=0,
            relief='flat',
        )
        style.map('TButton', 
            background=[('active', secondary_color), ('pressed', '#0056b3')]
        )
        
        # Checkbutton stili
        style.configure('Switch.TCheckbutton', 
            background=bg_color,
            indicatorbackground=bg_color,
            indicatorforeground=primary_color
        )
        
        # Combobox stili
        style.configure('TCombobox', 
            background=bg_color, 
            fieldbackground='white',
            font=('San Francisco', 13)
        )

class MultiTabApp:
    def __init__(self, root):
        self.root = root
        self.root.title("iOS Kamera Uygulaması")
        self.root.geometry("1400x900")
        
        # iOS tarzı tema uygula
        iOSStyleTheme(root)
        
        # Arka plan rengini iOS tarzı açık gri yap
        self.root.configure(background='#F2F2F7')
        
        # Yumuşak köşeli ana frame
        self.main_frame = tk.Frame(self.root, 
            background='#F2F2F7', 
            bd=0,
            highlightthickness=0
        )
        self.main_frame.pack(padx=10, pady=10, fill=tk.BOTH, expand=True)
        
        # Card tarzı container
        self.card_frame = tk.Frame(self.main_frame, 
            background='white', 
            bd=0,
            highlightthickness=0,
            relief='flat'
        )
        self.card_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        self.camera_threads = {}
        self.frame_queues = {}
        
        # Kamera bilgilerini al
        self.camera_info = self.get_camera_info()
        self.active_panels = {}
        
        self.create_main_frames()
        self.create_tabs()
        self.create_panels()
        self.update_all_panels()

    def get_camera_info(self):
        cameras = {}
        try:
            for i in range(5):  # Daha fazla kamera kontrolü için döngüyü genişletin
                cap = cv2.VideoCapture(i, cv2.CAP_DSHOW)
                if cap.isOpened():
                    cameras[f"Kamera {i}"] = i
                    cap.release()
        except Exception as e:
            print(f"Kamera kontrolü sırasında hata: {e}")
        return cameras if cameras else {"Kamera 0": 0}

    def create_main_frames(self):
        # iOS tarzı yumuşak köşeli notebook
        style = ttk.Style()
        style.configure('iOS.TNotebook', 
            background='#F2F2F7', 
            borderwidth=0, 
            relief='flat'
        )
        
        self.notebook = ttk.Notebook(self.card_frame, style='iOS.TNotebook')
        self.notebook.pack(expand=True, fill="both", padx=5, pady=5)
        
        self.bottom_frame = tk.Frame(self.card_frame, 
            background='white', 
            bd=0,
            highlightthickness=0
        )
        self.bottom_frame.pack(side=tk.BOTTOM, fill=tk.X, padx=5, pady=5)

    def create_tabs(self):
        # Ayarlar Tab'ı
        self.tab_settings = ttk.Frame(self.notebook)
        self.notebook.add(self.tab_settings, text="Ayarlar")
        self.create_settings_tab()
        
        # Liste Tab'ı
        self.tab_list = ttk.Frame(self.notebook)
        self.notebook.add(self.tab_list, text="Liste")
        self.create_list_tab()
        
        # Kontroller Tab'ı
        self.tab_controls = ttk.Frame(self.notebook)
        self.notebook.add(self.tab_controls, text="Kontroller")
        self.create_controls_tab()

    def create_settings_tab(self):
        frame = ttk.Frame(self.tab_settings)
        frame.pack(padx=10, pady=10, fill=tk.BOTH, expand=True)
        
        # Logo yükleme (isteğe bağlı)
        try:
            image_path = r'C:\Users\Tuna\Downloads\logo.png'
            img = Image.open(image_path)
            img = img.resize((200, 100), Image.LANCZOS)
            img_tk = ImageTk.PhotoImage(img)
            
            image_label = ttk.Label(frame, image=img_tk)
            image_label.image = img_tk
            image_label.pack(pady=10)
        except Exception as e:
            print(f"Görsel yüklenirken hata oluştu: {e}")
        
        # Parlaklık ayarı
        self.brightness = tk.DoubleVar(value=1.0)
        ttk.Label(frame, text="Parlaklık:").pack(anchor=tk.W)
        ttk.Scale(frame, from_=0.1, to=2.0, variable=self.brightness, 
                orient=tk.HORIZONTAL).pack(fill=tk.X)
        
        # Kontrast ayarı
        self.contrast = tk.DoubleVar(value=1.0)
        ttk.Label(frame, text="Kontrast:").pack(anchor=tk.W, pady=(10,0))
        ttk.Scale(frame, from_=0.1, to=3.0, variable=self.contrast, 
                orient=tk.HORIZONTAL).pack(fill=tk.X)

    def create_list_tab(self):
        frame = ttk.Frame(self.tab_list)
        frame.pack(padx=10, pady=10, fill=tk.BOTH, expand=True)
        
        self.listbox = tk.Listbox(frame)
        self.listbox.pack(fill=tk.BOTH, expand=True)
        
        for item in ["Kamera 1 Log", "Kamera 2 Log", "Sistem Mesajı", "Ayarlar Kaydedildi"]:
            self.listbox.insert(tk.END, item)

    def create_controls_tab(self):
        frame = ttk.Frame(self.tab_controls)
        frame.pack(padx=10, pady=10, fill=tk.BOTH, expand=True)
        
        ttk.Button(frame, text="Tüm Kayıtları Sil", command=self.clear_logs).pack(pady=5)
        ttk.Button(frame, text="Ayarları Kaydet", command=self.save_settings).pack(pady=5)
        ttk.Button(frame, text="Yenile", command=self.refresh).pack(pady=5)

    def create_panels(self):
        # iOS tarzı panel tasarımı
        self.camera_control_frame = tk.Frame(self.bottom_frame, 
            background='white')
        self.camera_control_frame.pack(side=tk.TOP, fill=tk.X, pady=10)
        
        self.camera_panel_frame = tk.Frame(self.bottom_frame, 
            background='white')
        self.camera_panel_frame.pack(side=tk.TOP, fill=tk.BOTH, expand=True)
        
        self.panels = {
            'original': {'title': 'Orijinal', 'column': 0, 'type': 'original'},
            'gray': {'title': 'Siyah-Beyaz', 'column': 1, 'type': 'gray'},
            'hsv': {'title': 'HSV', 'column': 2, 'type': 'hsv'},
            'noise': {'title': 'Gürültü Azaltma', 'column': 3, 'type': 'noise'}
        }
        
        for panel_id, info in self.panels.items():
            # iOS tarzı yumuşak köşeli çerçeve
            frame = tk.Frame(self.camera_panel_frame, 
                background='white', 
                bd=1,
                highlightthickness=1,
                highlightbackground='#E0E0E0',
                relief='flat')
            frame.grid(row=0, column=info['column'], padx=5, pady=5)
            
            # Modern, yuvarlatılmış canvas
            canvas = tk.Canvas(frame, 
                width=320, 
                height=240, 
                bg='#F2F2F7', 
                highlightthickness=0,
                bd=0,
                relief='flat')
            canvas.pack(pady=(10, 5))
            self.panels[panel_id]['canvas'] = canvas
            
            # iOS tarzı yazı tipi
            tk.Label(frame, 
                text=info['title'], 
                font=('San Francisco', 15, 'bold'),
                foreground='#007AFF',
                background='white'
            ).pack()
            
            control_frame = tk.Frame(frame, background='white')
            control_frame.pack(fill='x', pady=2)
            
            # iOS tarzı switch
            self.active_panels[panel_id] = tk.BooleanVar(value=False)
            chk = ttk.Checkbutton(
                control_frame, 
                text="Aktif", 
                style='Switch.TCheckbutton',
                variable=self.active_panels[panel_id],
                command=lambda pid=panel_id: self.toggle_camera(pid)
            )
            chk.pack(side=tk.LEFT, padx=5)
            
            # iOS tarzı combobox
            camera_var = tk.StringVar()
            camera_select = ttk.Combobox(
                control_frame, 
                textvariable=camera_var,
                values=list(self.camera_info.keys()),
                width=20, 
                state='readonly',
                style='TCombobox'
            )
            camera_select.pack(side=tk.LEFT, padx=5)
            self.panels[panel_id]['camera_var'] = camera_var
            
            if self.camera_info:
                camera_var.set(list(self.camera_info.keys())[0])
                camera_select.current(0)

            camera_select.bind('<<ComboboxSelected>>', 
                             lambda e, pid=panel_id: self.change_camera(pid))
            
            self.frame_queues[panel_id] = Queue(maxsize=2)

        # iOS tarzı butonlar
        ttk.Button(
            self.camera_control_frame, 
            text="Tümünü Başlat", 
            command=self.start_all
        ).pack(side=tk.LEFT, padx=5, pady=5)
        
        ttk.Button(
            self.camera_control_frame, 
            text="Tümünü Durdur", 
            command=self.stop_all
        ).pack(side=tk.LEFT, padx=5, pady=5)

    def update_all_panels(self):
        for panel_id, info in self.panels.items():
            if panel_id in self.camera_threads and not self.frame_queues[panel_id].empty():
                try:
                    _, frame = self.frame_queues[panel_id].get_nowait()
                    img = Image.fromarray(frame)
                    img_tk = ImageTk.PhotoImage(image=img)
                    info['canvas'].create_image(0, 0, anchor=tk.NW, image=img_tk)
                    info['canvas'].image = img_tk
                except:
                    pass
        self.root.after(30, self.update_all_panels)

    def change_camera(self, panel_id):
        if self.active_panels[panel_id].get():
            self.start_camera(panel_id)

    def start_all(self):
        for panel_id in self.panels:
            self.active_panels[panel_id].set(True)
            self.start_camera(panel_id)

    def stop_all(self):
        for panel_id in self.panels:
            self.active_panels[panel_id].set(False)
            self.stop_camera(panel_id)

    def clear_logs(self):
        self.listbox.delete(0, tk.END)

    def save_settings(self):
        print(f"Ayarlar kaydedildi - Parlaklık: {self.brightness.get()}, Kontrast: {self.contrast.get()}")

    def refresh(self):
        print("Sistem yenilendi")

    def __del__(self):
        self.stop_all()

if __name__ == "__main__":
    root = tk.Tk()
    app = MultiTabApp(root)
    root.mainloop()