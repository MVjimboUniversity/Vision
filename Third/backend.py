import os
import re
import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt
import scipy.fft as fft
from sklearn.neighbors import KNeighborsClassifier as Knn
from sklearn.decomposition import PCA
import mahotas as mt


def hist(img, param, use_plot=False):
    # img = cv.imread(path)
    img_c = cv.resize(img, (100, 100), interpolation=cv.INTER_LINEAR)
    histRange = (0, 256)
    accumulate = False
    hist = cv.calcHist([img_c], [0, 1, 2], None, [param, param, param], [0, 256, 0, 256, 0, 256], accumulate=accumulate).ravel()
    if use_plot:
        plt.hist(img.ravel(), param, histRange, rwidth=0.75)
    # else:
    #     hist = cv.calcHist([img], [0, 1, 2], None, [self.parameter], histRange, accumulate=accumulate).ravel()
    return hist


def pca(img, param, use_plot=False):
    # img = cv.imread(path, 0)
    img_c = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    img_c = cv.resize(img_c, (100, 100), interpolation= cv.INTER_LINEAR)
    pca_ = PCA(param)

    transform = pca_.fit_transform(img_c)

    return transform.ravel()


def haralick(img, param, use_plot=False):
    # img = cv.imread(path)
    # setting gaussian filter
    img_c = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    gaussian = mt.gaussian_filter(img_c, param)

    # setting threshold value
    gaussian = (gaussian > gaussian.mean())

    # making is labelled image
    labeled, n = mt.label(gaussian)

    # getting haralick features
    h_feature = mt.features.haralick(labeled).mean(axis=0)
    return h_feature


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
        0: hist,
        1: pca,
        2: haralick,
    }


def get_method(num):
    return Methods.get(num)


def predict(classifier, method_num, param, photo_path):
    method = get_method(method_num)
    photo = cv.imread(photo_path, 0)
    vec = method(photo, param)
    X = [vec]
    return classifier.predict(X)


def get_train_photos(path_to_db, photos):
    dirs = os.listdir(path_to_db)
    photos_list = os.listdir(os.path.join(path_to_db, dirs[0]))
    photos_dict = dict([(int(re.search(r'\d+', photo).group(0)) - 1, photo) for photo in photos_list])
    train_photos = [photos_dict[i] for i in photos]
    return train_photos


def get_test_photos(path_to_db, photos):
    dirs = os.listdir(path_to_db)
    photos_list = os.listdir(os.path.join(path_to_db, dirs[0]))
    photos_dict = dict([(int(re.search(r'\d+', photo).group(0)) - 1, photo) for photo in photos_list])
    test_photos = []
    for i, photo in photos_dict.items():
        if i not in photos:
            test_photos.append(photo)
    return test_photos


def count_error(pred, real):
    return np.sum(pred == real) / real.shape[0]


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
            photo = cv.imread(photo_path)
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
                photo = cv.imread(photo_path)
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
        photo = cv.imread(path)
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
                file_name = f".\\predict\\{str(label)}_{photo_name[:-5]}"
                plt.savefig(file_name)
                plt.clf()

    def get_plot(self, path):
        orig_class =int(path.split("\\")[2])
        photo = cv.imread(path)
        vec = self.method(photo, self.parameter)
        X = [vec]

        nearest = self.classifier.kneighbors(X, return_distance=False)
        nearest = nearest[0, 0]
        pred_class = nearest // self.n
        pred_num_photo = nearest % self.n

        dirs = os.listdir(self.path_to_db)
        pred_dir = dirs[pred_class]
        pred_class_name = int(pred_dir)
        pred_photo_name = self.photos[pred_num_photo]
        pred_path_photo = os.path.join(self.path_to_db, pred_dir, pred_photo_name)
        pred_photo = cv.imread(pred_path_photo)
        plt.subplot(121)
        plt.title(f"Original, {orig_class}")
        plt.imshow(cv.cvtColor(photo, cv.COLOR_BGR2RGB))
        plt.xticks([])
        plt.yticks([])
        plt.subplot(122)
        plt.title(f"Predicted, class {pred_class_name}")
        plt.imshow(cv.cvtColor(pred_photo, cv.COLOR_BGR2RGB))
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
            photo = cv.imread(path)
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
        self.get_plot(path, pred_vote)
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
                pred = self.predict(photo_path)
                self.get_plot(photo_path, pred)
                file_name = f".\\predict\\{str(label)}_{photo_name[:-5]}"
                plt.savefig(file_name)
                plt.clf()

    def get_plot(self, path, vote_pred):
        orig_class = int(path.split("\\")[2])
        photo = cv.imread(path)
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
            pred_class_name = int(pred_dir)
            pred_photo_name = self.photos[pred_num_photo]
            pred_path_photo = os.path.join(self.path_to_db, pred_dir, pred_photo_name)
            pred_photo = cv.imread(pred_path_photo)
            pred_photos.append(pred_photo)
            pred_class_names.append(pred_class_name)
        iter_photos = iter(pred_photos)
        iter_classes = iter(pred_class_names)
        plt.subplot(321)
        plt.title(f"Original, {orig_class}")
        plt.imshow(cv.cvtColor(photo, cv.COLOR_BGR2RGB), cmap='gray')
        plt.xticks([])
        plt.yticks([])
        plt.subplot(322)
        plt.title(f"Hist, class {next(iter_classes)}")
        plt.xticks([])
        plt.yticks([])
        plt.imshow(cv.cvtColor(next(iter_photos), cv.COLOR_BGR2RGB), cmap='gray')
        plt.subplot(323)
        plt.title(f"PCA, class {next(iter_classes)}")
        plt.xticks([])
        plt.yticks([])
        plt.imshow(cv.cvtColor(next(iter_photos), cv.COLOR_BGR2RGB), cmap='gray')
        plt.subplot(324)
        plt.title(f"Haralick, class {next(iter_classes)}")
        plt.xticks([])
        plt.yticks([])
        plt.imshow(cv.cvtColor(next(iter_photos), cv.COLOR_BGR2RGB), cmap='gray')
        # plt.subplot(325)
        # plt.title(f"Vote, class {vote_pred}")
        # plt.xticks([])
        # plt.yticks([])
        # plt.subplots_adjust(wspace=0.3, hspace=0.3)


if __name__ == '__main__':
    # lower_params = {
    #     0: 5,
    #     1: 10,
    #     2: 10,
    #     3: 5,
    #     4: 3
    # }
    # upper_params = {
    #     0: 15,
    #     1: 25,
    #     2: 25,
    #     3: 15,
    #     4: 10
    # }
    # classifier, params, precision = vote_fit_and_optimize('.\\orl_faces', lower_params, upper_params, [0, 1, 2, 5])
    # classifier.draw_all([0, 1, 2, 5])
    # fit_and_optimize('.\\orl_faces', 0, [5, 15], [0, 1, 2, 5])
    cl, p = fit(".\\pictures", 0, 20, range(14))
    cl.get_plot(".\\pictures\\1\\1.webp")
    # img = cv.imread
    # hist()
