from sklearn.metrics import precision_recall_fscore_support
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import BaggingClassifier
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.tree import DecisionTreeClassifier
from sklearn import svm
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from src.NeuralNetwork import NeuralNetwork
from sklearn.externals import joblib
from sklearn import metrics
import re

import src.tools as tools

def load_data(data_file):
    data = tools.read_csv_format(data_file)
    data = data[1:]
    def filter(var):
        return var[2:]
    data = list(map(filter,data))
    x,y = [],[]
    for var in data:
        x.append([float(var) for var in var[:-1]])
        y.append(int(var[-1]))
    return x,y

def svm_classify(X_train, y_train, X_test, y_test):
    clf = svm.SVC()
    clf.fit(X_train, y_train)
    pre_y_train = clf.predict(X_train)
    pre_y_test = clf.predict(X_test)
    report,confuse=analyse_predict_result(y_test,pre_y_test)
    return clf,report,confuse

def rf_classify(X_train, y_train, X_test, y_test):
    clf = RandomForestClassifier(random_state=0, n_estimators=500)
    clf.fit(X_train, y_train)
    pre_y_train = clf.predict(X_train)
    pre_y_test = clf.predict(X_test)
    report,confuse=analyse_predict_result(y_test,pre_y_test)
    return clf,report,confuse

def knn_classify(X_train, y_train, X_test, y_test):
    clf = KNeighborsClassifier(n_neighbors=5)
    clf.fit(X_train, y_train)
    pre_y_train = clf.predict(X_train)
    pre_y_test = clf.predict(X_test)
    report,confuse=analyse_predict_result(y_test,pre_y_test)
    return clf,report,confuse

def bagging_knn_classify(X_train, y_train, X_test, y_test):


    clf = BaggingClassifier(KNeighborsClassifier(),
                            max_samples=0.5, max_features=0.5)
    clf.fit(X_train, y_train)
    pre_y_test = clf.predict(X_test)
    report,confuse=analyse_predict_result(y_test,pre_y_test)
    return clf,report,confuse

def lr_classify(X_train, y_train, X_test, y_test):

    clf = LogisticRegression(C=1e5)
    clf.fit(X_train, y_train)
    pre_y_train = clf.predict(X_train)
    pre_y_test = clf.predict(X_test)
    report,confuse=analyse_predict_result(y_test,pre_y_test)
    return clf,report,confuse

def nb_classify(X_train, y_train, X_test, y_test):
    clf = GaussianNB()
    clf.fit(X_train, y_train)
    pre_y_train = clf.predict(X_train)
    pre_y_test = clf.predict(X_test)
    report,confuse=analyse_predict_result(y_test,pre_y_test)
    return clf,report,confuse

def da_classify(X_train, y_train, X_test, y_test):

    clf = QuadraticDiscriminantAnalysis()
    clf.fit(X_train, y_train)
    pre_y_test = clf.predict(X_test)
    report,confuse=analyse_predict_result(y_test,pre_y_test)
    return clf,report,confuse

def decisionTree_classify(X_train, y_train, X_test, y_test):
    clf = DecisionTreeClassifier(max_depth=5)
    clf.fit(X_train, y_train)
    pre_y_test = clf.predict(X_test)
    report,confuse=analyse_predict_result(y_test,pre_y_test)
    return clf,report,confuse

def GBDT_classify(X_train, y_train, X_test, y_test):
    from sklearn.ensemble import GradientBoostingClassifier
    clf = GradientBoostingClassifier(n_estimators=200)
    clf.fit(X_train, y_train)
    pre_y_test = clf.predict(X_test)
    report,confuse=analyse_predict_result(y_test,pre_y_test)
    return clf,report,confuse

def nn_classify(X_train,y_train,X_test,y_test):
    input_layer_nums = len(X_train[0])
    hidden_layer_nums = 5
    output_layer_nums = 1
    clf = NeuralNetwork(layers=[input_layer_nums,hidden_layer_nums,output_layer_nums])
    clf.fit(X_train,y_train)
    pre_y_test = clf.predict(X_test)
    # print(len(pre_y_test))
    # print(max(pre_y_test))
    # print(min(pre_y_test))
    report,confuse=analyse_predict_result(y_test,pre_y_test)
    return clf,report,confuse

def save_model(cls,save_path):
    joblib.dump(cls,save_path,compress=3)

def load_model(model_path):
    return joblib.load(model_path)


models = {
    "randomforest":rf_classify,
    "svm":svm_classify,
    "naivebayes":nb_classify,
    # "nn":nn_classify
}

def train_model(train_file,test_file,model_name,model_save_path=None,res_save_path =None):
    train_x,train_y = load_data(train_file)
    test_x,test_y = load_data(test_file)

    if model_name in models.keys():
        _model = models[model_name]
    elif model_name in models.values():
        _model = model_name
    clf,report,confuse = _model(train_x,train_y,test_x,test_y)
    if model_save_path!=None:
        save_model(clf,model_save_path)
    string = str(confuse[0][0])+","+str(confuse[0][1])+","+str(confuse[1][0])+","+str(confuse[1][1])+ "\n"
    string2 = ""
    for var in report:
        string2+= ','.join(list(map(str,var)))+'\n'
    if res_save_path!=None:
        tools.save_txt(string+string2,res_save_path)
    return string+string2

def predict(model_path,x_predict,y_predict = None):
    _model = load_model(model_path)
    result= _model.predict(x_predict)
    return result

def analyse_predict_result(real_y,predict_y):
    report = metrics.classification_report(real_y,predict_y)
    confuse =metrics.confusion_matrix(real_y,predict_y)
    report = [re.split(" {2,20}",var.strip()) for var in report.split("\n")[2:] if var.strip()!=""]
    return report,confuse




if __name__ == "__main__":
    train_files = "../res/feature_data/train/东方园林.csv"
    test_files = "../res/feature_data/test/东方园林.csv"
    model_path = "../res/model/东方园林/randomforest.m"
    result = train_model(train_file=train_files,test_file=test_files,model_name="randomforest",model_save_path=model_path)
    print(result)
    x_test,y_test = load_data(test_files)
    pre_y_test = predict(model_path,x_test)
    report,confuse = analyse_predict_result(y_test,pre_y_test)


    print(report,confuse)

