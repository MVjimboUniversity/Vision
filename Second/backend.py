import os
import re
import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt
import scipy.fft as fft
from sklearn.neighbors import KNeighborsClassifier as Knn


def histogram(img, histSize, use_plot=False):
    histRange = (0, 256)
    accumulate = False
    if use_plot:
        hist, bars, _ = plt.hist(img.ravel(), histSize, histRange, rwidth=0.75)
    else:
        hist = cv.calcHist([img], [0], None, [histSize], histRange, accumulate=accumulate).ravel()
    return hist


def dft(img, p, use_plot=False):
    f = np.fft.fft2(img, (p, p))
    fshift = np.fft.fftshift(f)
    magnitude_spectrum = np.abs(fshift)
    # magnitude_spectrum = magnitude_spectrum[p // 2:, :]
    # magnitude_spectrum = cv.normalize(np.abs(fshift), None, 0, 255, cv.NORM_MINMAX)
    if use_plot:
        plt.imshow(magnitude_spectrum, cmap='gray')
    return magnitude_spectrum.ravel()


def dft1(img, p, use_plot=False):
    f = np.fft.fft2(img, (p, p))
    f[0, 0] = 0
    fshift = np.fft.fftshift(f)
    magnitude_spectrum = np.abs(fshift)
    # magnitude_spectrum = cv.normalize(np.abs(fshift), None, 0, 255, cv.NORM_MINMAX)
    # height_mid = magnitude_spectrum.shape[0] // 2
    # width_mid = magnitude_spectrum.shape[1] // 2
    # p_mid = p // 2
    # print(magnitude_spectrum[height_mid , width_mid - 1])
    if use_plot:
        plt.imshow(magnitude_spectrum[p//2:, :], cmap='gray')
    return magnitude_spectrum.ravel()


def dct(img, p, use_plot=False):
    spectrum = fft.dct(fft.dct(img.T, n=p, type=2, norm='ortho').T, n=p, type=2, norm='ortho')
    # spectrum[0, 0] = 0
    # spectrum = np.abs(spectrum)
    # spectrum = cv.normalize(spectrum, None, 0, (img.shape[0]*img.shape[1])**(1/2), cv.NORM_MINMAX)
    # spectrum_vector = [spectrum[i, j] for i in range(p) for j in range(0, p-i)]
    if use_plot:
        plt.imshow(spectrum, cmap='gray')
    return spectrum.ravel()
    # return np.array(spectrum_vector)


def scale(img, scale_height, use_plot=False):
    scale_width = int(scale_height * (img.shape[1] / img.shape[0]))
    scaled_img = cv.resize(img, (scale_width, scale_height), interpolation=cv.INTER_LINEAR)
    if use_plot:
        plt.imshow(scaled_img, cmap='gray')
    return scaled_img.ravel()


def grad(img, h, use_plot=False):
    d = []
    for i in range(h, img.shape[0]-h):
        res = np.sum(img[i:i+h, :].astype(np.int16) - img[i-h:i, :].astype(np.int16))
        d.append(res)
    d = np.array(d)
    if use_plot:
        plt.plot(d)
        plt.grid()
    return d


Methods = {
        0: histogram,
        1: dft,
        2: dct,
        3: scale,
        4: grad
    }


def get_method(num):
    return Methods.get(num)


def predict(classifier, method_num, param, photo_path):
    method = get_method(method_num)
    photo = cv.imread(photo_path, 0)
    vec = method(photo, param)
    X = [vec]
    return classifier.predict(X)


def count_error(pred, real):
    return np.sum(pred == real) / real.shape[0]


def get_test_photos(path_to_db, photos):
    dirs = os.listdir(path_to_db)
    photos_list = os.listdir(os.path.join(path_to_db, dirs[0]))
    photos_dict = dict([(int(re.search(r'\d+', photo).group(0)) - 1, photo) for photo in photos_list])
    test_photos = []
    for i, photo in photos_dict.items():
        if i not in photos:
            test_photos.append(photo)
    return test_photos


def count_precision(classifier, path_to_db, photos):
    y = []
    pred_all = []
    dirs = os.listdir(path_to_db)
    test_photos = get_test_photos(path_to_db, photos)
    for dir in dirs:
        label = int(re.search(r'\d+', dir).group(0))
        for photo_name in test_photos:
            photo_path = os.path.join(path_to_db, dir, photo_name)
            pred = classifier.predict(photo_path)
            pred_all.append(pred)
            y.append(label)

    pred_all = np.array(pred_all)
    y = np.array(y)
    precision = count_error(pred_all, y)
    return precision


def fit(path_to_db, method_num, param, photos):
    classifier_base = Knn(1)
    method = get_method(method_num)
    dirs = os.listdir(path_to_db)
    X = []
    y = []
    photos_list = os.listdir(os.path.join(path_to_db, dirs[0]))
    photos_dict = dict([(int(re.search(r'\d+', photo).group(0)) - 1, photo) for photo in photos_list])
    train_photos = [photos_dict[i] for i in photos]
    for dir in dirs:
        label = int(re.search(r'\d+', dir).group(0))
        for photo_name in train_photos:
            photo_path = os.path.join(path_to_db, dir, photo_name)
            photo = cv.imread(photo_path, 0)
            vec = method(photo, param)
            X.append(vec)
            y.append(label)
    X = np.array(X)
    y = np.array(y)
    classifier_base = classifier_base.fit(X, y)
    classifier = Classifier(classifier_base, method, param, train_photos, path_to_db)
    precision = count_precision(classifier, path_to_db, photos)
    return classifier, precision


def fit_and_optimize(path_to_db, method_num, params, photos, draw=False):
    best_param = params[0]
    max_precision = 0.
    precision_arr = []
    best_classifier = None
    for param in range(params[0], params[1]+1):
        classifier, precision = fit(path_to_db, method_num, param, photos)
        precision_arr.append(precision)
        if max_precision < precision:
            best_param = param
            best_classifier = classifier
            max_precision = precision
    if draw:
        fig, ax = plt.subplots()
        ax.set_title('Точность классификатора в зависимости от параметра')
        ax.plot(range(params[0], params[1]+1), precision_arr)
        ax.grid()
        ax.set_xlabel('Значение параметра')
        ax.set_ylabel('Точность')
        plt.savefig('precision.png')
        precision_img = cv.imread('precision.png', cv.IMREAD_COLOR)
        cv.imshow("Precision", precision_img)
    return best_classifier, best_param, max_precision


def vote_fit(path_to_db, params, photos):
    dirs = os.listdir(path_to_db)
    photos_list = os.listdir(os.path.join(path_to_db, dirs[0]))
    photos_dict = dict([(int(re.search(r'\d+', photo).group(0)) - 1, photo) for photo in photos_list])
    train_photos = [photos_dict[i] for i in photos]

    classifiers = {}

    for method_num, method in Methods.items():
        classifier = Knn(1)
        param = params[method_num]
        X = []
        y = []
        for dir in dirs:
            label = int(re.search(r'\d+', dir).group(0))
            for photo_name in train_photos:
                photo_path = os.path.join(path_to_db, dir, photo_name)
                photo = cv.imread(photo_path, 0)
                vec = method(photo, param)
                X.append(vec)
                y.append(label)
        X = np.array(X)
        y = np.array(y)
        classifiers.update({method_num: classifier.fit(X, y)})
    classifier = VoteClassifier(classifiers, params, train_photos, path_to_db)
    precision = count_precision(classifier, path_to_db, photos)
    return classifier, precision


def vote_fit_and_optimize(path_to_db, lower_params, upper_params, photos):
    best_classifiers = {}
    best_params = {}
    train_photos = None
    for method_num, method in Methods.items():
        params = (lower_params[method_num], upper_params[method_num])
        classifier, param, precision = fit_and_optimize(path_to_db, method_num, params, photos)
        best_classifiers.update({method_num: classifier.classifier})
        best_params.update({method_num: param})
        if train_photos is None:
            train_photos = classifier.photos

    classifier = VoteClassifier(best_classifiers, best_params, train_photos, path_to_db)

    precision = count_precision(classifier, path_to_db, photos)
    return classifier, best_params, precision


class Classifier:
    def __init__(self, classifier, method, parameter, photos, path_to_db):
        self.classifier = classifier
        self.method = method
        self.parameter = parameter
        self.n = len(photos)
        self.photos = photos
        self.path_to_db = path_to_db

    def fit(self, X, y):
        self.classifier = self.classifier.fit(X, y)
        return self

    def predict(self, path, show_nearest=False):
        photo = cv.imread(path, 0)
        vec = self.method(photo, self.parameter)
        X = [vec]
        pred = self.classifier.predict(X)[0]
        if show_nearest:
            self.draw_predict(path)
        return pred

    def draw_predict(self, path):
        self.get_plot(path)
        plt.savefig("predict.png")
        img = cv.imread("predict.png", cv.IMREAD_COLOR)
        cv.imshow("Predict", img)

    def draw_all(self, photos):
        dirs = os.listdir(self.path_to_db)
        test_photos = get_test_photos(self.path_to_db, photos)
        for dir in dirs:
            label = int(re.search(r'\d+', dir).group(0))
            for photo_name in test_photos:
                photo_path = os.path.join(self.path_to_db, dir, photo_name)
                self.get_plot(photo_path)
                file_name = f".\\predict\\{str(label)}_{photo_name[:-4]}"
                plt.savefig(file_name)
                plt.clf()

    def get_plot(self, path):
        photo = cv.imread(path, 0)
        vec = self.method(photo, self.parameter)
        X = [vec]

        nearest = self.classifier.kneighbors(X, return_distance=False)
        nearest = nearest[0, 0]
        pred_class = nearest // self.n
        pred_num_photo = nearest % self.n

        dirs = os.listdir(self.path_to_db)
        pred_dir = dirs[pred_class]
        pred_class_name = int(pred_dir[1:])
        pred_photo_name = self.photos[pred_num_photo]
        pred_path_photo = os.path.join(self.path_to_db, pred_dir, pred_photo_name)
        pred_photo = cv.imread(pred_path_photo, 0)
        plt.subplot(121)
        plt.title(f"Original")
        plt.imshow(photo, cmap='gray')
        plt.xticks([])
        plt.yticks([])
        plt.subplot(122)
        plt.title(f"Predicted, class {pred_class_name}")
        plt.imshow(pred_photo, cmap='gray')
        plt.xticks([])
        plt.yticks([])


class VoteClassifier:
    def __init__(self, classifiers, parameters, photos, path_to_db):
        self.classifiers = classifiers
        self.methods = Methods.copy()
        self.parameters = parameters
        self.n = len(photos)
        self.photos = photos
        self.path_to_db = path_to_db

    def fit(self, X, y):
        for classifier in self.classifiers.values():
            classifier.fit(X, y)
        return self

    def predict(self, path, show_nearest=False):
        pred_all = []
        for i, classifier in self.classifiers.items():
            photo = cv.imread(path, 0)
            method = self.methods[i]
            param = self.parameters[i]
            vec = method(photo, param)
            X = [vec]
            pred = classifier.predict(X)[0]
            pred_all.append(pred)

        # Возвращает предсказанный класс
        pred_vote = np.argmax(np.bincount(pred_all))
        if show_nearest:
            self.draw_predict(path, pred_vote)
        return pred_vote

    def draw_predict(self, path, pred_vote):
        self.get_plot(path)
        plt.savefig("predict.png")
        img = cv.imread("predict.png", cv.IMREAD_COLOR)
        cv.imshow("Predict", img)

    def draw_all(self, photos):
        dirs = os.listdir(self.path_to_db)
        test_photos = get_test_photos(self.path_to_db, photos)
        for dir in dirs:
            label = int(re.search(r'\d+', dir).group(0))
            for photo_name in test_photos:
                photo_path = os.path.join(self.path_to_db, dir, photo_name)
                self.get_plot(photo_path)
                file_name = f".\\predict\\{str(label)}_{photo_name[:-4]}"
                plt.savefig(file_name)
                plt.clf()

    def get_plot(self, path):
        photo = cv.imread(path, 0)
        pred_photos = []
        pred_class_names = []
        for method, param, classifier in zip(self.methods.values(), self.parameters.values(),
                                             self.classifiers.values()):
            vec = method(photo, param)
            X = [vec]

            nearest = classifier.kneighbors(X, return_distance=False)
            nearest = nearest[0, 0]
            pred_class = nearest // self.n
            pred_num_photo = nearest % self.n

            dirs = os.listdir(self.path_to_db)
            pred_dir = dirs[pred_class]
            pred_class_name = int(pred_dir[1:])
            pred_photo_name = self.photos[pred_num_photo]
            pred_path_photo = os.path.join(self.path_to_db, pred_dir, pred_photo_name)
            pred_photo = cv.imread(pred_path_photo, 0)
            pred_photos.append(pred_photo)
            pred_class_names.append(pred_class_name)
        iter_photos = iter(pred_photos)
        iter_classes = iter(pred_class_names)
        plt.subplot(231)
        plt.title(f"Original")
        plt.imshow(photo, cmap='gray')
        plt.xticks([])
        plt.yticks([])
        plt.subplot(232)
        plt.title(f"Hist, class {next(iter_classes)}")
        plt.xticks([])
        plt.yticks([])
        plt.imshow(next(iter_photos), cmap='gray')
        plt.subplot(233)
        plt.title(f"DFT, class {next(iter_classes)}")
        plt.xticks([])
        plt.yticks([])
        plt.imshow(next(iter_photos), cmap='gray')
        plt.subplot(234)
        plt.title(f"DCT, class {next(iter_classes)}")
        plt.xticks([])
        plt.yticks([])
        plt.imshow(next(iter_photos), cmap='gray')
        plt.subplot(235)
        plt.title(f"Scale, class {next(iter_classes)}")
        plt.xticks([])
        plt.yticks([])
        plt.imshow(next(iter_photos), cmap='gray')
        plt.subplot(236)
        plt.title(f"Grad, class {next(iter_classes)}")
        plt.xticks([])
        plt.yticks([])
        plt.imshow(next(iter_photos), cmap='gray')


if __name__ == '__main__':
    lower_params = {
        0: 5,
        1: 10,
        2: 10,
        3: 5,
        4: 3
    }
    upper_params = {
        0: 15,
        1: 25,
        2: 25,
        3: 15,
        4: 10
    }
    classifier, params, precision = vote_fit_and_optimize('.\\orl_faces', lower_params, upper_params, [0, 1, 2, 5])
    classifier.draw_all([0, 1, 2, 5])
    # fit_and_optimize('.\\orl_faces', 0, [5, 15], [0, 1, 2, 5])
