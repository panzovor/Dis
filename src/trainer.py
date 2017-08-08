import os
import src.tools as tools
import subprocess
import nltk as nltk

project_dir ="D:/workspace/Dis"
java_path=os.environ.get('JAVA_HOME')+"\\bin\\java.exe"
weka_path="../res/parameter/weka.jar"
_cmd =  [java_path,"-cp", weka_path]
def execCmd(cmd):
   sub=subprocess.Popen(cmd,shell=True,stdout=subprocess.PIPE)
   stdout,stderror = sub.communicate()
   # print("out",stdout)
   # print("error",stderror)
   return None

train_parameter ={
    "RandomForest":"weka.classifiers.trees.RandomForest -I 10 -K 0 -S 1",
    "NaiveBayes":"weka.classifiers.bayes.NaiveBayes",
    "RBF":"weka.classifiers.functions.RBFNetwork -B 2 -S 1 -R 1.0E-8 -M -1 -W 0.1",
    "SMO":"weka.classifiers.functions.SMO -C 1.0 -L 0.001 -P 1.0E-12 -N 0 -V -1 -W 1 -K \"weka.classifiers.functions.supportVector.PolyKernel -C 250007 -E 1.0\"",
    "MultilayerPerceptron":"weka.classifiers.functions.MultilayerPerceptron -L 0.3 -M 0.2 -N 500 -V 0 -S 0 -E 20 -H a",
    "BayesNet":"weka.classifiers.bayes.BayesNet -D -Q weka.classifiers.bayes.net.search.local.K2 -- -P 1 -S BAYES -E weka.classifiers.bayes.net.estimate.SimpleEstimator -- -A 0.5"
}

predict_parameter ={
    "RandomForest": "java weka.classifiers.trees.RandomForest",
    "NaiveBayes": "java weka.classifiers.bayes.NaiveBayes",
    "RBF": "java weka.classifiers.functions.RBFNetwork",
    "SMO": "java weka.classifiers.functions.SMO",
    "MultilayerPerceptron": "java weka.classifiers.functions.MultilayerPerceptron",
    "BayesNet": "java weka.classifiers.bayes.BayesNet"

}

def predict(model_name,model_file, test_file,class_index):
    if model_name in predict_parameter.keys():
        if not os.path.exists(model_file):
            print("model file doesnot exist")
            return None
        if not os.path.exists(test_file):
            print("test file doesnot exist")
            return None
        try:
            cmd = predict_parameter[model_name] +" -p "+str(class_index)+" -l "+model_file+" -T "+test_file
            cmd = cmd.replace("..",project_dir)
            print(cmd)
            result = execCmd(cmd)
            return result
        except:
            print("check your model file or test file")
    else:
        print("wrong model name")

def train(model_name,train_file, save_file):
    # print(save_file)
    if model_name in train_parameter:
        if not os.path.exists(train_file):
            print("train file doesnot exist")
            return None
        try:
            cmd = _cmd+[train_parameter[model_name].replace("..",project_dir)," -t ",train_file," -d ",save_file]
            print(cmd)
            result = execCmd(cmd)
            # return result
        except:
            print("wrong trainfile")
    else:
        print("wrong model name")



def train_model(train_file,save_root):
    models_names = ['MultilayerPerceptron', 'RBF', 'NaiveBayes', 'RandomForest', 'SMO', 'BayesNet']
    for name in models_names:
        train(name,train_file,save_root+"/"+name+".model")
    return save_root

def test_model(models_root,test_file,save_root):

    models_names = ['MultilayerPerceptron', 'RBF', 'NaiveBayes', 'RandomForest', 'SMO', 'BayesNet']
    # test_file = "../res/train_featuredata/"+test_name+".arff_data"
    for name in models_names:
        save_file = save_root+name+".txt"
        model_file = models_root+name+".model"
        result =predict(name,model_file,test_file,class_index=16)
        tools.save_txt(result,save_file)
    # return save_root


if __name__ == "__main__":
    # train_nltk(arff_file="../res/arff_data/新希望.csv.arff")
    # print(java_path)
    # print(weka_path)
    train_model(train_file="../res/arff_data/train/大西洋.csv.arff",save_root="../res/model/大西洋/")
    print(['C:\\Program Files\\Java\\jdk1.8.0_65\\bin\\java.exe', '-cp', '../res/parameter/weka.jar', 'weka.classifiers.trees.RandomForest', '-p', '15', '-l', '../res/model/大西洋/NaiveBayes.model', '-T', '../res/arff_data/train/大西洋.csv.arff'])