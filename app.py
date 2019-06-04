#!/usr/bin/env python3

import sys
import cv2
import glob
import logging
import numpy as np
import tkinter as tk
import tkinter.messagebox as pop_msg
import os
from tkinter import ttk
import tkinter.filedialog as filedialog
from PIL import Image, ImageTk
import threading as td
import subprocess
from threading import RLock
from PIL.ImageTk import PhotoImage

from enhancer import Enhancer


# if u wanna running it on CPU.
# os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

class APP(tk.Tk):

    def __init__(self):
        super().__init__(className='fantasticFilter')

        ''' ========== Locks ========== '''

        self.resize_lock = td.Lock()
        self.resizing = False

        ''' ======= Tk widgets ======== '''
        self.style = ttk.Style()
        self.frame_main_left = None
        self.frame_main_center = None
        self.frame_main_right = None
        self.canvas: ResizingCanvas = None
        self.enhance_pb = None
        self.start_enhance_btn = None
        self.input_resize_height = None
        self.input_resize_width = None

        ''' ========= Tk flags ======== '''
        self._model_path_obj = tk.StringVar(self)
        self.main_right_model_label = tk.StringVar(self)
        self.main_right_model_label.set("使用模型：<無>")
        self._vignette_scale = 1
        self._vignette_should_update = False
        self._vignette_lock = RLock()

        ''' ======== neuronal ========= '''
        self._model = Enhancer()

        ''' ===== internal flags ====== '''
        self._model_loaded = self._model.model_available
        self._image_loaded = lambda: self._main_image_origin is not None
        self.status_text = tk.StringVar(self)
        self.resize_width = tk.StringVar(self)
        self.resize_width.trace("w", self._resize_width_listener)
        self.resize_height = tk.StringVar(self)
        self.resize_height.trace("w", self._resize_height_listener)
        self.status_text.set("就緒")
        ''' ===== np array images ===== '''
        self._main_image_origin = None
        self._main_image_current_clean = None
        self._main_image_enhanced = None

        ''' ========== theme ========== '''
        '''
        # MacOS is beautiful enough so that theme is not required. 
        try:
            from ttkthemes import ThemedStyle
            style = ThemedStyle(self)
            style.set_theme("arc")
        except ImportError as e:
            class ThemedStyle:
                pass
        '''
        ''' ====== configuration ====== '''

        self.model_dir = resource_path("pretrained/")
        self.vignette_handler()

    def enhance_listener(self, *args):
        if not (self._check_image() and self._check_model()):
            return
        thread = td.Thread(target=self._enhance_task)
        self.status_text.set("增強圖片中..")
        self.enhance_pb.start()
        try:
            self.start_enhance_btn.config(state="disabled")
            self.config(cursor='wait')
        except Exception as e:
            logging.warning(str(e))
        thread.start()
        self._enhance_handler(thread)

    def _resize_height_listener(self, *args):
        if not self._check_image(): return
        with self.resize_lock:
            if self.resizing:
                return
            self.resizing = True
        origin_height, origin_width, _ = np.shape(self._main_image_origin)
        resize_height = 0 if not self.resize_height.get() else int(self.resize_height.get())
        ratio = resize_height / origin_height
        new_width = int(origin_width * ratio)
        self.resize_width.set(new_width)
        with self.resize_lock:
            self.resizing = False

    def _resize_width_listener(self, *args):
        if not self._check_image(): return
        with self.resize_lock:
            if self.resizing:
                return
            self.resizing = True
        origin_height, origin_width, _ = np.shape(self._main_image_origin)
        resize_width = 0 if not self.resize_width.get() else int(self.resize_width.get())
        ratio = resize_width / origin_width
        new_height = int(origin_height * ratio)
        self.resize_height.set(new_height)
        with self.resize_lock:
            self.resizing = False

    def _enhance_task(self):
        new_height = int(self.resize_height.get())
        new_width = int(self.resize_width.get())
        new_height = new_height if new_height is not 0 else 1
        new_width = new_width if new_width is not 0 else 1

        resize_image = cv2.resize(self._main_image_origin, dsize=(new_width, new_height))
        resize_image = resize_image[new_height % 8:, new_width % 8:, :]
        self._main_image_enhanced = self._model.sample(resize_image, denoise=False)
        self._main_image_current_clean = self._main_image_enhanced
        print(self._main_image_enhanced)

    def _enhance_handler(self, thread: td.Thread):
        if thread.is_alive():
            self.after(100, lambda: self._enhance_handler(thread))
        else:
            self.enhance_pb.stop()
            self.start_enhance_btn.config(state="normal")
            self.config(cursor='')
            image = Image.fromarray(np.asarray(self._main_image_enhanced))
            self.canvas.set_main_image(image)
            self.canvas.request_update()
            self.status_text.set("處理完成！")
            try:
                subprocess.check_output(["notify-send", "圖片處理完成！", "<b>幻想濾鏡™</b>處理好圖片囉！", "--icon=face-glasses"])
            except FileNotFoundError:
                print("can't send notification.")
            self.after(3000, lambda: self.status_text.set("就緒"))

    def open_image_listener(self, *args):

        try:
            filename = subprocess.check_output(['zenity', '--file-selection']).decode("utf-8").strip()
        except FileNotFoundError:
            filename = filedialog.askopenfilename()
        except subprocess.CalledProcessError:
            filename = False

        if not filename:
            logging.info("cancel opening image.")
            return False
        try:
            logging.info("open image:", filename)
            image = Image.open(filename)
            self._main_image_origin = np.asarray(image)
            self.canvas.set_main_image(image)
            self.canvas.image = image
            self._main_image_current_clean = self._main_image_origin
            self.canvas.request_update()

            self.resize_height.set(image.height)
            self.resize_width.set(image.width)



        except IOError as e:
            logging.error("open image failed!")
            logging.error(str(e))
            return False

    def select_model_listener(self, *kwargs):

        model_name = self._model_path_obj.get()
        model_path = self._get_model_path(model_name)
        self.main_right_model_label.set("使用模型：" + model_name)
        if not os.path.isfile(model_path):
            model_path = filedialog.askopenfilename(filetypes=(("pre-trained _model", "*.pb"), ("all files", "*.*")))
        if not os.path.isfile(model_path):
            return False

        t_init_model = td.Thread(target=self.init_model, args=(model_path,))
        t_init_model.start()

        waiting_win = tk.Toplevel(self)
        waiting_frame = ttk.Frame(waiting_win)
        waiting_frame.pack(fill='both', expand=True, side='top')

        waiting_win.lift(aboveThis=self)
        waiting_win.geometry("300x130")
        waiting_win.resizable(0, 0)
        waiting_win.transient(self)
        waiting_win.grab_set()
        waiting_win.protocol("WM_DELETE_WINDOW", lambda: None)
        waiting_win.wm_title("Loading Pre-trained Model")

        ttk.Label(waiting_frame, text="\nloading '" + model_name + "' ... \nThis won't take long.\n\n").pack(side='top')

        waiting_win.pb = ttk.Progressbar(waiting_frame, length=200, mode="indeterminate", orient=tk.HORIZONTAL)
        waiting_win.pb.pack(pady=5)
        waiting_win.pb.start()

        self.load_model_waiting_win(waiting_win, t_init_model)

    def load_model_waiting_win(self, waiting_win: tk.Toplevel, load_thread: td.Thread):
        if load_thread.is_alive():
            waiting_win.after(100, lambda: self.load_model_waiting_win(waiting_win, load_thread))
        else:
            waiting_win.pb.stop()
            waiting_win.destroy()

    def init_model(self, path: str):
        try:
            self._model.load_model(model_path=path)
        except Exception as e:
            pop_msg.showerror("Something went wrong.. ", "無法載入模型！")
            logging.error(str(e))

    def vignette_listener(self, value):

        self._vignette_scale = 2. - float(value)
        with self._vignette_lock:
            self._vignette_should_update = True

    def vignette_handler(self):
        with self._vignette_lock:
            if self._vignette_should_update:
                logging.info("update vignette")
                if not self._check_image():
                    return False
                else:
                    if self._vignette_scale >= 1.99:
                        result_image = self._main_image_current_clean
                    else:
                        image = self._main_image_current_clean.copy()
                        result_image = self.vignette(image, scale=self._vignette_scale)
                result_image_pil = Image.fromarray(result_image)
                self.canvas.set_main_image(result_image_pil)
                self._vignette_should_update = False

        self.after(24, self.vignette_handler)

    def save(self, *args):
        if not self._check_image():
            return
        path = filedialog.asksaveasfilename(initialfile='enhanced', defaultextension='.png',
                                            filetypes=[("PNG Images", "*.png"), ("JPEG Files", "*.jpg")])
        if path:
            image = self.canvas.main_image
            image.save(path)

    def run(self):

        self.title("Fantastic Filter")

        self.geometry("%dx%d+50+40" % (800, 500))

        """
        menu_bar = tk.Menu(self, background='#aaa')

        menu_file = tk.Menu(menu_bar, tearoff=0)
        menu_edit = tk.Menu(menu_bar, tearoff=0)
        menu_help = tk.Menu(menu_bar, tearoff=0)
        menu_bar.add_cascade(label='檔案', menu=menu_file)
        menu_bar.add_cascade(label='編輯', menu=menu_edit)
        menu_bar.add_cascade(label='說明', menu=menu_help)

        ''' Menu-File '''
        menu_file.add_command(label='開啟影像', command=self.open_image_listener)
        menu_file.add_command(label='離開', command=self.quit)

        ''' Menu-Edit '''
        # TODO: add menu-edit items
        ''' Menu-Help '''
        menu_help.add_command(label='致謝')
        """

        ''' toolbar '''

        frame_toolbar = ttk.Frame(self)
        frame_toolbar.pack(fill='x')
        open_image_btn = ttk.Button(frame_toolbar, text="開啟圖片", command=self.open_image_listener)
        open_image_btn.pack(side='left', padx=5, pady=8)

        select_model_label = ttk.Label(frame_toolbar, text="請選擇模型：")
        select_model_label.pack(side='left', padx=(10, 0))
        model_cbb = ttk.Combobox(frame_toolbar, textvariable=self._model_path_obj)  # 初始化
        model_cbb.pack(side='left')
        model_cbb["values"] = ['請選擇模型'] + list(map(self._get_model_name, self._get_model_list())) + ['選擇其他模型..']
        if len(model_cbb["values"]) > 0:
            model_cbb.current(0)  # 選擇第一個
        model_cbb.bind("<<ComboboxSelected>>", self.select_model_listener)  # 绑定事件,(下拉列表框被选中时)

        ''' main area '''

        # split into 3 part, | load _model,image... | image preview | edit.. save.|

        frame_main = ttk.Frame(self)
        frame_main.pack(fill='both', expand=True, side='bottom')
        '''
        self.frame_main_left = ttk.Frame(frame_main)
        self.frame_main_left.grid(row=0, column=0, sticky="nsew")
        '''
        self.frame_main_center = ttk.Frame(frame_main, width=600)
        self.frame_main_center.grid(row=0, column=0, sticky='news')
        bg = self.style.lookup('TFrame', 'background')
        bg = "#e7e7e7" if bg == 'systemWindowBody' else bg
        self.canvas = ResizingCanvas(self.frame_main_center, bg=bg, bd=0, highlightthickness=0, relief='ridge')
        self.canvas.pack(fill='both', expand=True, pady=10, padx=10)

        self.frame_main_right = ttk.Frame(frame_main, width=200)
        self.frame_main_right.grid(row=0, column=1, sticky='news', padx=20, pady=20)

        frame_fantastic = ttk.Frame(self.frame_main_right)
        frame_fantastic.pack(fill='x')
        ttk.Label(frame_fantastic, textvariable=self.main_right_model_label).pack(fill='x', pady=5)
        self.start_enhance_btn = ttk.Button(frame_fantastic, text="開始處理", command=self.enhance_listener)
        self.start_enhance_btn.pack(fill='x', expand=True)

        self.enhance_pb = ttk.Progressbar(frame_fantastic, length=160, mode="indeterminate", orient=tk.HORIZONTAL)
        self.enhance_pb.pack(fill='x', pady=5)

        ttk.Separator(self.frame_main_right, orient='horizontal').pack(fill='x', pady=10)

        frame_resize = ttk.Frame(self.frame_main_right)
        frame_resize.pack()

        ttk.Label(frame_resize, text="寬/高").pack(fill='x')
        frame_resize_inputs = ttk.Frame(frame_resize)
        frame_resize_inputs.pack()
        self.input_resize_height = ttk.Entry(frame_resize_inputs, textvariable=self.resize_height, validate='key',
                                             validatecommand=(self.register(isnumeric_or_blank), "%P"), width=9)
        self.input_resize_width = ttk.Entry(frame_resize_inputs, textvariable=self.resize_width, validate='key',
                                            validatecommand=(self.register(isnumeric_or_blank), "%P"), width=9)
        self.input_resize_width.pack(side='left')
        self.input_resize_height.pack(side='right')

        ttk.Separator(self.frame_main_right, orient='horizontal').pack(fill='x', pady=10)

        frame_controls_vignette = ttk.Frame(self.frame_main_right)
        # frame_controls_vignette.pack(fill='x')

        controls_vignette_label = ttk.Label(frame_controls_vignette, text='暈影')
        controls_vignette_label.pack(fill='x')
        controls_vignette_scale = ttk.Scale(frame_controls_vignette, length=160, command=self.vignette_listener)
        controls_vignette_scale.pack(fill='x', ipadx=5)

        # ttk.Separator(self.frame_main_right, orient='horizontal').pack(fill='x', pady=10)

        frame_save = ttk.Frame(self.frame_main_right)
        frame_save.pack(fill='x', pady=10)
        ttk.Button(frame_save, text='儲存', command=self.save).pack(fill='x', expand=True)

        tk.Grid.rowconfigure(frame_main, 0, weight=1)
        tk.Grid.columnconfigure(frame_main, 0, weight=1)
        self.style.configure('gary.TFrame', background="#eaeaea")
        self.style.configure('gary.TLabel', background="#eaeaea")
        frame_status = ttk.Frame(frame_main, style='gary.TFrame')
        frame_status.grid(row=2, column=0, columnspan=2, sticky="nwes")

        status_bar = ttk.Label(frame_status, textvariable=self.status_text, style='gary.TLabel')
        status_bar.pack(side='left', padx=5, pady=0)
        # self.config(menu=menu_bar)

        self.bind_all("<Command-o>", self.open_image_listener)
        self.bind_all("<Control-o>", self.open_image_listener)
        self.bind_all("<Command-s>", self.save)
        self.bind_all("<Control-s>", self.save)
        self.mainloop()

    @staticmethod
    def vignette(image, scale):

        # every numpy array has a shape tuple
        width, height = image.shape[:2]

        xs = np.arange(width)
        ys = np.arange(height)
        distance_squared = (xs - width / 2.0)[..., np.newaxis] ** 2 + (ys - height / 2.0) ** 2

        sigma_squared = (width / 2.0) ** 2 + (height / 2.0) ** 2
        sigma_squared /= 2
        mask = np.exp(-distance_squared / sigma_squared)

        # the easiest way to control the strength of the mask

        # the easiest way to control the strength of the mask

        scale_revers = 1 / scale

        new_h = int(mask.shape[0] * scale_revers)
        new_w = int(mask.shape[1] * scale_revers)
        y_start = int((mask.shape[0] - new_h) / 2)
        x_start = int((mask.shape[1] - new_w) / 2)

        mask_new = mask.copy()[x_start:new_h, y_start:new_w]
        mask_new = cv2.resize(mask_new, (mask.shape[1], mask.shape[0]))

        result = image * mask_new[..., np.newaxis]
        return np.uint8(result)

    def _check_model(self):
        if not self._model_loaded():
            pop_msg.showinfo("Hmm..", "請先選擇模型！")
            return False
        else:
            return True

    def _check_image(self):
        if not self._image_loaded():
            pop_msg.showinfo("Umm..", "請先開啟圖片！")
            return False
        else:
            return True

    def _get_model_list(self):
        base_dir = self.model_dir
        return glob.glob(base_dir + "/*.pb")

    @staticmethod
    def _get_model_name(path):
        return os.path.basename(path.replace(".pb", ''))

    def _get_model_path(self, name):
        return self.model_dir + '/' + name + ".pb"


def isnumeric_or_blank(my_str: str):
    return my_str.isnumeric() or my_str == ''


class ResizingCanvas(tk.Canvas):
    image: PhotoImage

    def __init__(self, parent, **args):
        super().__init__(parent, args)
        self.main_image = None
        self.width = self.winfo_width()
        self.height = self.winfo_height()
        self.lock = RLock()
        self._should_update = False
        self.bind("<Configure>", self.on_resize)
        self._update_handler()

        self.main_image_tk = None
        self.image_pi = None

    def _update_handler(self, interval=200):
        with self.lock:
            if self._should_update:
                self.update_now()
                self._should_update = False
        self.after(interval, lambda: self._update_handler(interval))

    def on_resize(self, event):
        self.request_update()
        # self.update()

    def request_update(self):
        with self.lock:
            self._should_update = True

    def update_now(self, *args):
        self.width = self.winfo_width()
        self.height = self.winfo_height()

        image = self.main_image
        if image is not None:
            w, h = image.size
            scale = min((self.width / w), (self.height / h))
            image_resize = image.resize((int(w * scale), (int(h * scale))), Image.ANTIALIAS)
            self.image_pi = ImageTk.PhotoImage(image=image_resize)

            # self.delete("all")
            self.main_image_tk = self.create_image(self.width / 2, self.height / 2, anchor='center',
                                                   image=self.image_pi)
            self.image = self.image_pi
            self.update()

    def set_main_image(self, image: Image):

        if self.image_pi is not None:
            w, h = image.size
            scale = min((self.width / w), (self.height / h))
            image_resize = image.resize((int(w * scale), (int(h * scale))), Image.ANTIALIAS)
            self.image = image_resize
            self.image_pi = ImageTk.PhotoImage(image=image_resize)
            self.itemconfigure(self.main_image_tk, image=self.image_pi)

        # self.image = image
        self.main_image = image


def resource_path(relative_path):
    if hasattr(sys, '_MEIPASS'):
        print(sys._MEIPASS)
        return sys._MEIPASS + '/' + relative_path
    return os.path.join(os.path.dirname(os.path.abspath(__file__)), relative_path)


if __name__ == '__main__':
    app = APP()
    app.run()
