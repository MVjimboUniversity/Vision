import os
import re
import tkinter as tk
from tkinter import filedialog
import tkinter.messagebox as mb
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
from backend import fit, fit_and_optimize, vote_fit_and_optimize, vote_fit


class Root(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title('Классификатор')
        self.frame_train = FrameTrain(self)
        # self.frame_radio = FrameRadio(self)
        # self.frame_buttons = FrameButtons(self)
        self.frame_train.pack()
        # self.frame_radio.pack(expand=True)
        # self.frame_buttons.pack(expand=True)


class FrameTrain(tk.Frame):
    def __init__(self, root):
        super().__init__(root)

        self.path_to_db = '.\\orl_faces'

        self.method_names = {
            0: 'Гистограмма',
            1: 'DFT',
            2: 'DCT',
            3: 'Scale',
            4: 'Градиент',
        }

        self.method_lower_bound = {
            0: tk.IntVar(),
            1: tk.IntVar(),
            2: tk.IntVar(),
            3: tk.IntVar(),
            4: tk.IntVar()
        }

        self.method_upper_bound = {
            0: tk.IntVar(),
            1: tk.IntVar(),
            2: tk.IntVar(),
            3: tk.IntVar(),
            4: tk.IntVar()
        }

        self.method_params = {
            0: tk.IntVar(),
            1: tk.IntVar(),
            2: tk.IntVar(),
            3: tk.IntVar(),
            4: tk.IntVar()
        }

        self.opt_params = {
            0: [27, 27, 28, 27, 27, 23, 17, 17, 17],
            1: [35, 35, 35, 35, 32, 32, 32, 33, 30],
            2: [30, 30, 28, 28, 28, 27, 27, 27, 27],
            3: [22, 22, 24, 24, 24, 22, 25, 22, 22],
            4: [8, 7, 6, 6, 6, 6, 7, 5, 5]
        }

        self.num_train = []

        self.method = tk.IntVar()
        self.method.set(0)

        self.optimize = tk.IntVar()
        self.optimize.set(0)

        self.classifier = None

        self.image_path = None

        self.photo = None
        self.counter = 0
        self.UI()

    def UI(self):
        train_set_frame = tk.Frame(self)
        train_frame = tk.Frame(train_set_frame)
        label_test = tk.Label(train_frame, text='Выбрать изображения для выборки')
        list = tk.Listbox(train_frame, selectmode=tk.MULTIPLE, height=5)
        list.insert(0, *[i for i in range(1, 11)])
        save_btn = tk.Button(train_frame, text='Сохранить выбор',
                             command=lambda: self.save_test(list.curselection(), label_choosed_train))
        label_choosed_train = tk.Label(train_frame)
        label_test.grid(row=0, column=0)
        list.grid(row=1, column=0)
        save_btn.grid(row=2, column=0)
        label_choosed_train.grid(row=3, column=0)
        train_frame.grid(row=0, column=0)

        methods_frame = tk.Frame(train_set_frame)
        label_method = tk.Label(methods_frame, text='Методы')
        hist = tk.Radiobutton(methods_frame, text="Гистограмма", variable=self.method, value=0)
        dft = tk.Radiobutton(methods_frame, text="DFT", variable=self.method, value=1)
        dct = tk.Radiobutton(methods_frame, text="DCT", variable=self.method, value=2)
        scale = tk.Radiobutton(methods_frame, text="Scale", variable=self.method, value=3)
        grad = tk.Radiobutton(methods_frame, text="Градиент", variable=self.method, value=4)
        vote = tk.Radiobutton(methods_frame, text="Система голосования", variable=self.method, value=5)
        label_method.grid(row=0, column=0, sticky=tk.W)
        hist.grid(row=1, column=0, sticky=tk.W)
        dft.grid(row=1, column=1, sticky=tk.W)
        dct.grid(row=2, column=0, sticky=tk.W)
        scale.grid(row=2, column=1, sticky=tk.W)
        grad.grid(row=3, column=0, sticky=tk.W)
        vote.grid(row=3, column=1, sticky=tk.W)
        methods_frame.grid(row=1, column=0)

        parameters_type_frame = tk.Frame(train_set_frame)
        label_parameters = tk.Label(parameters_type_frame, text='Параметры')
        intervals = tk.Radiobutton(parameters_type_frame, text="Интервалы", variable=self.optimize, value=0)
        values = tk.Radiobutton(parameters_type_frame, text="Значение", variable=self.optimize, value=1)
        opt_values = tk.Radiobutton(parameters_type_frame, text="Оптимальное значение", variable=self.optimize, value=2)
        label_parameters.grid(row=0, column=0, sticky=tk.W)
        intervals.grid(row=1, column=0, sticky=tk.W)
        values.grid(row=1, column=1, sticky=tk.W)
        opt_values.grid(row=2, column=0, columnspan=2)
        choose_btn = tk.Button(parameters_type_frame, text='Задать параметры',
                               command=lambda: self.set_parameters(parameters_frame, self.method.get(),
                                                                   self.optimize.get()))
        choose_btn.grid(row=3, column=0, columnspan=2, pady=5)
        parameters_type_frame.grid(row=2, column=0)

        parameters_frame = tk.Frame(train_set_frame)
        parameters_frame.grid(row=3, column=0)

        train_frame = tk.Frame(train_set_frame)
        train_btn = tk.Button(train_frame, text='Обучить',
                               command=lambda: self.fit(train_label))
        train_label = tk.Label(train_frame)
        train_btn.grid(row=0, column=0, pady=5)
        train_label.grid(row=1, column=0)
        train_frame.grid(row=4, column=0)

        test_frame = tk.Frame(train_set_frame)
        b1 = tk.Button(test_frame, text="Выбрать изображение", command=self.get_image_name)
        label = tk.Label(test_frame)
        b2 = tk.Button(test_frame, text="Распознать", command=lambda: self.classification(label))
        auto_b2 = tk.Button(test_frame, text="Автоматическое распознование",
                            command=lambda: self.auto_classification(dynamic_show_frame))
        b3 = tk.Button(test_frame, text="Закрыть окна", command=close)
        b1.grid(row=0, column=0, pady=5)
        b2.grid(row=1, column=0, pady=5)
        auto_b2.grid(row=2, column=0, pady=5)
        label.grid(row=3, column=0, pady=5)
        b3.grid(row=4, column=0, pady=5)
        test_frame.grid(row=5, column=0)

        dynamic_show_frame = tk.Frame(self)
        # scroll_y = tk.Scrollbar(dynamic_show_frame, orient=tk.VERTICAL)
        # canvas = tk.Canvas(dynamic_show_frame, height=600, width=700, yscrollcommand=scroll_y.set)
        # scroll_y.config(command=canvas.yview)
        # canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=tk.TRUE)
        # scroll_y.pack(fill=tk.Y, side=tk.RIGHT, expand=tk.FALSE)
        # self.photo = tk.PhotoImage(file='.\\predict\\10_10.png')
        # canvas.create_image(0, 0, anchor='nw', image=self.photo)
        # c_photo = canvas.create_image(0, 0, anchor='nw', image=photo_tk)

        train_set_frame.pack(side=tk.LEFT)
        dynamic_show_frame.pack(side=tk.RIGHT)

    def save_test(self, selected, label):
        self.num_train = selected
        label.config(text=f"Выбранные фото {','.join(map(str, [i+1 for i in selected]))}")
        # dirs = os.listdir(self.path_to_db)
        # photos_list = os.listdir(os.path.join(self.path_to_db, dirs[0]))
        # photos_dict = dict([(int(re.search(r'\d+', photo).group(0)) - 1, photo) for photo in photos_list])
        # train_photos = [photos_dict[i] for i in selected]
        # photo_all_arr = []
        # for dir in dirs:
        #     photo_line_arr = []
        #     for photo in train_photos:
        #         photo_path = os.path.join(self.path_to_db, dir, photo)
        #         photo = cv.imread(photo_path, 0)
        #         photo = cv.resize(photo, None, fx=0.5, fy=0.5, interpolation=cv.INTER_LINEAR)
        #         photo_line_arr.append(photo)
        #     photo_line = np.concatenate(photo_line_arr, axis=0)
        #     photo_all_arr.append(photo_line)
        # photo_all = np.concatenate(photo_all_arr, axis=1)
        # cv.imshow("Examples", photo_all)

    def set_parameters(self, frame, method, optimize):
        for widget in frame.winfo_children():
            widget.destroy()
        if method != 5:
            methods = [method]
        else:
            methods = range(5)
        if optimize == 0:
            labels = [tk.Label(frame, text=self.method_names.get(method)) for method in methods]
            entries_low = [tk.Entry(frame, textvariable=self.method_lower_bound[i], width=3)
                           for i in methods]
            entries_high = [tk.Entry(frame, textvariable=self.method_upper_bound[i], width=3)
                            for i in methods]
            for i, (label, entry_low, entry_high) in enumerate(zip(labels, entries_low, entries_high)):
                label.grid(row=i, column=0, sticky=tk.W)
                tk.Label(frame, text='от').grid(row=i, column=1)
                entry_low.grid(row=i, column=2)
                tk.Label(frame, text='до').grid(row=i, column=3)
                entry_high.grid(row=i, column=4)
        elif optimize == 1:
            labels_opt = [tk.Label(frame, text=self.method_names.get(method)) for method in methods]
            entries_opt = [tk.Entry(frame, textvariable=self.method_params[i], width=3)
                           for i in methods]
            for i, (label, entry) in enumerate(zip(labels_opt, entries_opt)):
                label.grid(row=i, column=0, sticky=tk.W)
                entry.grid(row=i, column=1)

    def fit(self, label):
        if self.method.get() != 5:
            if self.optimize.get() == 0:
                method = self.method.get()
                lower_bound = self.method_lower_bound[method].get()
                upper_bound = self.method_upper_bound[method].get()
                self.classifier, best_param, precision = fit_and_optimize(self.path_to_db, method,
                                                   (lower_bound, upper_bound), self.num_train, True)
                self.method_params[method].set(best_param)
            elif self.optimize.get() == 1:
                method = self.method.get()
                param = self.method_params[method].get()
                self.classifier, precision = fit(self.path_to_db, method, param, self.num_train)
            elif self.optimize.get() == 2:
                method = self.method.get()
                param = self.opt_params[method][len(self.num_train)-1]
                print(param)
                self.classifier, precision = fit(self.path_to_db, method, param, self.num_train)
        else:
            if self.optimize.get() == 0:
                lower_params = dict([(i, param.get()) for i, param in self.method_lower_bound.items()])
                upper_params = dict([(i, param.get()) for i, param in self.method_upper_bound.items()])
                self.classifier, best_params, precision = vote_fit_and_optimize(self.path_to_db,
                                                                               lower_params, upper_params,
                                                                               self.num_train)
                for method, param in best_params.items():
                    self.method_params[method].set(param)
            elif self.optimize.get() == 1:
                params = dict([(i, param.get()) for i, param in self.method_params.items()])
                self.classifier, precision = vote_fit(self.path_to_db, params, self.num_train)
            elif self.optimize.get() == 2:
                method = self.method.get()
                params = dict([(i, param[len(self.num_train)-1]) for i, param in self.opt_params.items()])
                self.classifier, precision = vote_fit(self.path_to_db, params, self.num_train)
        label.config(text=f'Лучшая точность - {precision:.3f}')

    def get_image_name(self):
        try:
            image_path = filedialog.askopenfilename(initialdir="./", title="Выберете изображение",
                                                    filetypes=(("Изображения(*.jpg, *.png, *.pgm)", "*.jpg *.png *.pgm"),))
        except Exception:
            self.image_path = None
        else:
            self.image_path = image_path
            img = cv.imread(image_path, cv.IMREAD_COLOR)
            cv.imshow("Selected image", img)

    def classification(self, label):
        if self.image_path is None:
            self.show_error('Не задано изображение.')
        else:
            pred = self.classifier.predict(self.image_path, True)
            label.config(text=f'Предсказанный класс - {pred}')

    def auto_classification(self, frame):
        for photo in os.listdir('.\\predict'):
            os.remove(os.path.join('.\\predict', photo))
        self.classifier.draw_all(self.num_train)
        self.counter = 0
        photo_name = os.listdir('.\\predict')[self.counter]
        photo_path = os.path.join('.\\predict', photo_name)
        self.photo = tk.PhotoImage(file=photo_path)
        for widget in frame.winfo_children():
            widget.destroy()
        next_btn = tk.Button(frame, text='Следующая', command=lambda: self.show_next(img_label, next_btn,
                                                                                    prev_btn))
        prev_btn = tk.Button(frame, text='Предыдущая', command=lambda: self.show_prev(img_label, next_btn,
                                                                                    prev_btn))
        img_label = tk.Label(frame, image=self.photo)
        prev_btn.grid(row=0, column=0)
        next_btn.grid(row=0, column=1)
        img_label.grid(row=1, column=0, columnspan=2)

        prev_btn.grid_remove()

    def show_next(self, label, next_btn, prev_btn):
        self.counter += 1
        dirs = os.listdir('.\\predict')
        n = len(dirs)
        photo_name = dirs[self.counter]
        photo_path = os.path.join('.\\predict', photo_name)
        self.photo = tk.PhotoImage(file=photo_path)
        if self.counter == n-1:
            next_btn.grid_remove()
        prev_btn.grid()
        label.config(image=self.photo)

    def show_prev(self, label, next_btn, prev_btn):
        self.counter -= 1
        dirs = os.listdir('.\\predict')
        photo_name = dirs[self.counter]
        photo_path = os.path.join('.\\predict', photo_name)
        self.photo = tk.PhotoImage(file=photo_path)
        if self.counter == 0:
            prev_btn.grid_remove()
        next_btn.grid()
        label.config(image=self.photo)

    @staticmethod
    def show_error(msg):
        mb.showerror("Ошибка.", msg)


def close():
    cv.destroyAllWindows()


def main():
    root = Root()
    root.mainloop()


if __name__ == '__main__':
    main()
