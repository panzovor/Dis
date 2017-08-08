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
    clf_res ="SVM Metrics : {0}".format(precision_recall_fscore_support(y_test, pre_y_test))
    return clf,clf_res

def rf_classify(X_train, y_train, X_test, y_test):
    clf = RandomForestClassifier(random_state=0, n_estimators=500)
    clf.fit(X_train, y_train)
    pre_y_train = clf.predict(X_train)
    pre_y_test = clf.predict(X_test)
    clf_res ="rf Metrics : {0}".format(precision_recall_fscore_support(y_test, pre_y_test))
    return clf, clf_res

def knn_classify(X_train, y_train, X_test, y_test):
    clf = KNeighborsClassifier(n_neighbors=5)
    clf.fit(X_train, y_train)
    pre_y_train = clf.predict(X_train)
    pre_y_test = clf.predict(X_test)
    print("knn Metrics : {0}".format(precision_recall_fscore_support(y_test, pre_y_test)))

def bagging_knn_classify(X_train, y_train, X_test, y_test):


    clf = BaggingClassifier(KNeighborsClassifier(),
                            max_samples=0.5, max_features=0.5)
    clf.fit(X_train, y_train)
    pre_y_test = clf.predict(X_test)
    print("bagging_knn Metrics : {0}".format(precision_recall_fscore_support(y_test, pre_y_test)))

def lr_classify(X_train, y_train, X_test, y_test):

    clf = LogisticRegression(C=1e5)
    clf.fit(X_train, y_train)
    pre_y_train = clf.predict(X_train)
    pre_y_test = clf.predict(X_test)
    print("lr Metrics : {0}".format(precision_recall_fscore_support(y_test, pre_y_test)))

def nb_classify(X_train, y_train, X_test, y_test):
    clf = GaussianNB()
    clf.fit(X_train, y_train)
    pre_y_train = clf.predict(X_train)
    pre_y_test = clf.predict(X_test)
    clf_res ="nb Metrics : {0}".format(precision_recall_fscore_support(y_test, pre_y_test))
    return clf,clf_res

def da_classify(X_train, y_train, X_test, y_test):

    clf = QuadraticDiscriminantAnalysis()
    clf.fit(X_train, y_train)
    pre_y_test = clf.predict(X_test)
    print("da Metrics : {0}".format(precision_recall_fscore_support(y_test, pre_y_test)))

def decisionTree_classify(X_train, y_train, X_test, y_test):
    clf = DecisionTreeClassifier(max_depth=5)
    clf.fit(X_train, y_train)
    pre_y_test = clf.predict(X_test)
    print("DT Metrics : {0}".format(precision_recall_fscore_support(y_test, pre_y_test)))

def GBDT_classify(X_train, y_train, X_test, y_test):
    from sklearn.ensemble import GradientBoostingClassifier
    clf = GradientBoostingClassifier(n_estimators=200)
    clf.fit(X_train, y_train)
    pre_y_test = clf.predict(X_test)
    print("GBDT Metrics : {0}".format(precision_recall_fscore_support(y_test, pre_y_test)))

def nn_classify(X_train,y_train,X_test,y_test):
    input_layer_nums = len(X_train[0])
    hidden_layer_nums = 5
    output_layer_nums = 1
    nn = NeuralNetwork(layers=[input_layer_nums,hidden_layer_nums,output_layer_nums])
    nn.fit(X_train,y_train)
    pre_y_test = nn.predict(X_test)
    print(len(pre_y_test))
    print(max(pre_y_test))
    print(min(pre_y_test))
    clf_res ="NN Metrics : {0}".format(precision_recall_fscore_support(y_test, pre_y_test))
    return nn,clf_res

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

    clf,result = _model(train_x,train_y,test_x,test_y)
    if model_save_path!=None:
        save_model(clf,model_save_path)
    if res_save_path!=None:
        tools.save_txt(result,res_save_path)
    return result

def predict(model_path,x_predict,y_predict = None):
    _model = load_model(model_path)
    result= _model.predict(x_predict)
    return result

def analyse_predict_result(real_y,predict_y):
    report = metrics.classification_report(real_y,predict_y)
    confuse =metrics.confusion_matrix(real_y,predict_y)
    return report,confuse




if __name__ == "__main__":
    train_files = "../res/feature_data/train/广深铁路.csv"
    test_files = "../res/feature_data/test/广深铁路.csv"
    model_path = "../res/model/广深铁路/randomforest.m"
    result = train_model(train_file=train_files,test_file=test_files,model_name="svm",model_save_path=model_path)
    print(result)
    x_test,y_test = load_data(test_files)
    pre_y_test = predict(model_path,x_test)
    report,confuse = analyse_predict_result(y_test,pre_y_test)
    print(report,confuse)

