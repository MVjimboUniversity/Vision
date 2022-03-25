import tkinter as tk
from tkinter import filedialog
import tkinter.messagebox as mb
import cv2 as cv


class Root(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title('Задание 1')
        self.frame_radio = FrameRadio(self)
        frame = FrameButtons(self)
        self.frame_radio.pack(expand=True)
        frame.pack(expand=True)

    def get_method(self):
        return self.frame_radio.get_method()


class FrameRadio(tk.Frame):
    def __init__(self, root):
        super().__init__(root)

        self.method = tk.IntVar()
        self.method.set(0)
        self.UI()

    def UI(self):
        labelTM = tk.Label(self, text='Методы Template Matching:')
        labelVJ = tk.Label(self, text='Метод Виолы-Джонса:')
        sqdiff = tk.Radiobutton(self, text="TM SQDIFF", variable=self.method, value=0)
        sqdiff_norm = tk.Radiobutton(self, text="TM SQDIFF NORMED", variable=self.method, value=1)
        tm_ccorr = tk.Radiobutton(self, text="TM CCORR", variable=self.method, value=2)
        tm_ccorr_norm = tk.Radiobutton(self, text="TM CCORR NORMED", variable=self.method, value=3)
        tm_coeff = tk.Radiobutton(self, text="TM CCOEFF", variable=self.method, value=4)
        tm_coeff_norm = tk.Radiobutton(self, text="TM CCOEFF NORMED", variable=self.method, value=5)
        vj = tk.Radiobutton(self, variable=self.method, value=6)
        labelTM.grid(row=0, column=0, sticky=tk.W)
        sqdiff.grid(row=1, column=0, sticky=tk.W)
        sqdiff_norm.grid(row=1, column=1, sticky=tk.W)
        tm_ccorr.grid(row=2, column=0, sticky=tk.W)
        tm_ccorr_norm.grid(row=2, column=1, sticky=tk.W)
        tm_coeff.grid(row=3, column=0, sticky=tk.W)
        tm_coeff_norm.grid(row=3, column=1, sticky=tk.W)
        labelVJ.grid(row=4, column=0, sticky=tk.W)
        vj.grid(row=4, column=1, sticky=tk.W)

    def get_method(self):
        return self.method.get()


class FrameButtons(tk.Frame):
    def __init__(self, root):
        super().__init__(root)
        self.root = root
        self.image_path = None
        self.template_path = None
        self.UI()

    def UI(self):
        b1 = tk.Button(self, text="Выбрать изображение", command=self.get_image_name)
        b1.grid(row=0, column=2, pady=5)
        b2 = tk.Button(self, text="Выбрать шаблон", command=self.get_template_name)
        b2.grid(row=1, column=2, pady=5)
        b3 = tk.Button(self, text="Поиск", command=self.matching)
        b3.grid(row=2, column=2, pady=5)
        b3 = tk.Button(self, text="Закрыть окна", command=close)
        b3.grid(row=3, column=2, pady=5)

    def get_image_name(self):
        try:
            image_path = filedialog.askopenfilename(initialdir="./", title="Выберете изображение",
                                                    filetypes=(("Изображения(*.jpg, *.png)", "*.jpg *.png"),))
        except Exception:
            self.image_path = None
        else:
            self.image_path = image_path
            img = cv.imread(image_path, cv.IMREAD_COLOR)
            cv.imshow("Image", img)

    def get_template_name(self):
        try:
            template_path = filedialog.askopenfilename(initialdir="./", title="Выберете шаблон",
                                                       filetypes=(("Изображения(*.jpg, *.png)", "*.jpg *.png"),))
        except Exception:
            self.template_path = None
        else:
            self.template_path = template_path
            template = cv.imread(template_path, cv.IMREAD_COLOR)
            cv.imshow("Template", template)

    def get_method(self):
        return self.root.get_method()

    @staticmethod
    def show_error(msg):
        mb.showerror("Ошибка", msg)

    def matching(self):
        method = self.get_method()
        if self.image_path is None and self.template_path is None and method != 6:
            FrameButtons.show_error('Изображение и шаблон не заданы')
        elif self.image_path is None:
            FrameButtons.show_error('Изображение не задано')
        elif self.template_path is None and method != 6:
            FrameButtons.show_error('Шаблон не задан')
        else:
            if method == 6:
                result = matchingVJ(self.image_path)
                cv.imshow("Result", result)
            else:
                result, image = matchingTM(self.image_path, self.template_path, method)
                cv.imshow("Result", result)
                cv.imshow("Face", image)


def close():
    cv.destroyAllWindows()


def matchingVJ(image_path):
    img = cv.imread(image_path)
    img_display = img.copy()
    img_gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

    cascade_path = "C:\\anaconda3\envs\zrenie\Library\etc\haarcascades\haarcascade_frontalface_default.xml"
    face_cascade = cv.CascadeClassifier(cascade_path)

    faces = face_cascade.detectMultiScale(img_gray, scaleFactor=1.3, minNeighbors=5)

    for (x, y, w, h) in faces:
        cv.rectangle(img_display, (x, y), (x+w, y+h), (0, 0, 128), 2)

    return img_display


def matchingTM(image_path, template_path, method):
    img = cv.imread(image_path, cv.IMREAD_COLOR)
    templ = cv.imread(template_path, cv.IMREAD_COLOR)
    img_display = img.copy()

    result = cv.matchTemplate(img, templ, method)

    cv.normalize(result, result, 0, 1, cv.NORM_MINMAX, -1)

    _minVal, _maxVal, minLoc, maxLoc = cv.minMaxLoc(result, None)

    if method == cv.TM_SQDIFF or method == cv.TM_SQDIFF_NORMED:
        matchLoc = minLoc
    else:
        matchLoc = maxLoc

    cv.rectangle(img_display, matchLoc, (matchLoc[0] + templ.shape[1], matchLoc[1] + templ.shape[0]), (0, 0, 128), 2, 8, 0)
    cv.rectangle(result, matchLoc, (matchLoc[0] + templ.shape[1], matchLoc[1] + templ.shape[0]), (0, 0, 128), 2, 8, 0)

    return result, img_display


def main():
    root = Root()
    root.mainloop()


if __name__ == '__main__':
    main()
