import loaddata
import numpy as np
from sklearn import svm
from PIL import Image
from PIL import ImageFilter
import time
import sklearn
from sklearn.model_selection import StratifiedKFold
import joblib
import matplotlib.pyplot as plt


def preprocess_data(data):
    my_data = []
    for i in range(data.shape[0]):
        img = Image.fromarray(np.array(data[i]))
        img_ = img.resize([50, 50])
        img_ = img_.filter(ImageFilter.SMOOTH)
        img_ = img_.filter(ImageFilter.SMOOTH)
        img_ = img_.filter(ImageFilter.SMOOTH)
        img_ = img_.filter(ImageFilter.SMOOTH)
        my_data.append(np.array(img_).reshape(2500).tolist())
    my_data = np.array(my_data)
    return my_data


def train_svm():
    data = np.load('data_for_other.npy', allow_pickle=True)
    labels = np.load('labels_for_other.npy')
    permutation = np.random.permutation(labels.shape[0])
    data = data[permutation]
    labels = labels[permutation]
    my_data = preprocess_data(data)
    scalar = sklearn.preprocessing.StandardScaler().fit(my_data)
    my_data = scalar.transform(my_data)
    joblib.dump(scalar, 'std_scaler.bin')
    genC = StratifiedKFold(n_splits=10)
    model = svm.SVC(C=1.0, cache_size=200, class_weight=None, coef0=0.0, decision_function_shape='ovo', degree=3,
                    gamma='auto', kernel='rbf', max_iter=-1, probability=False, random_state=None, shrinking=True,
                    tol=0.001, verbose=False)
    # # model = svm.SVC(decision_function_shape='ovo', gamma=0.0001)
    start = time.time()
    i = 0
    print("start training")
    score = 0
    for train, validation in genC.split(my_data, labels):
        i = i + 1
        train_data = my_data[train]
        train_labels = labels[train]
        valid_data = my_data[validation]
        valid_labels = labels[validation]
        model.fit(train_data, train_labels)
        score_n = model.score(valid_data, valid_labels)
        print(i, '  score:', score_n)
        # score_x = model.score(letters, l)
        # print('sb', score_x)
        if score_n > score:
            index = validation
            joblib.dump(model, "SVM_model.sav",compress=3)
            score = score_n

    # model = GridSearchCV(svm.SVC(kernel='rbf'), param_grid=params, cv=3)

    end = time.time()
    # y_hat = model.predict(my_data)
    print("Training Time: ", end - start)
    print('score:', score)


def test_svm(data):
    data = preprocess_data(data)
    scalar = joblib.load('std_scaler.bin')
    data = scalar.transform(data)
    model = joblib.load('SVM_model.sav')
    res = model.predict(data)
    return res


# letters = []
# l = []
# for i in range(labels.size):
#     if labels[i] == 1:
#         letters.append(data[i])
#         l.append(labels[i])
# letters = np.array(letters)
# letters = SVM_load.preprocess_data(letters)
# letters = sklearn.preprocessing.scale(letters)

# permutation = np.random.permutation(labels.shape[0])
# data = data[permutation]
# labels = labels[permutation]
# my_data = SVM_load.preprocess_data(data)
# scalar = sklearn.preprocessing.StandardScaler().fit(my_data)
#
# my_data = scalar.transform(my_data)
# joblib.dump(scalar, 'std_scaler.bin')
# genC = StratifiedKFold(n_splits=10)
# model = svm.SVC(C=1.0, cache_size=200, class_weight=None, coef0=0.0, decision_function_shape='ovo', degree=3,
#                 gamma='auto', kernel='rbf', max_iter=-1, probability=False, random_state=None, shrinking=True,
#                 tol=0.001, verbose=False)
# # # model = svm.SVC(decision_function_shape='ovo', gamma=0.0001)
# start = time.time()
# i = 0
# print("start training")
# score = 0

# letters = []
# l = []
# for i in range(labels1.size):
#     if labels1[i] == 1:
#         letters.append(data1[i])
#         l.append(labels1[i])
# letters = np.array(letters)
# letters = SVM_load.preprocess_data(letters)
# letters = scalar.transform(letters)
# for train, validation in genC.split(my_data, labels):
#     i = i+1
#     train_data = my_data[train]
#     train_labels = labels[train]
#     valid_data = my_data[validation]
#     valid_labels = labels[validation]
#     model.fit(train_data, train_labels)
#     score_n = model.score(valid_data, valid_labels)
#     print(i, '  score:', score_n)
#     # score_x = model.score(letters, l)
#     # print('sb', score_x)
#     if score_n > score:
#         index = validation
#         joblib.dump(model, "SVM_model.sav")
#         score = score_n
#
# # model = GridSearchCV(svm.SVC(kernel='rbf'), param_grid=params, cv=3)
#
# end = time.time()
# # y_hat = model.predict(my_data)
# print("Training Time: ", end - start)
# print('score:', score)

# data = loaddata.load_pkl('train_data.pkl')
# res = SVM_load.my_test_other(data)
# print(res)


# data = np.load('test_data.npy')
# result = test_svm(data)
# print('总数', result.size)
# print('正确', sum(result == -1))


