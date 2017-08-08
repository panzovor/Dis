from nltk.internals import java,config_java
import subprocess
from sys import stdin
import re

### weka cmd:
### java weka.classifiers.trees.J48 -p 9 -l directory-path\bank.model -T directory-path \bank-new.arff
weka_class_path = "../res/parameter/weka.jar"
config_java()


### input
def weka_classify(arff_file,model_file):
    class_index=1
    if model_file =="":
        return None
    with open(arff_file,mode="r",encoding="utf-8") as file:
        lines = file.readlines()
        for i in range(lines.__len__()):
            if "@attribute class" in lines[i]:
                class_index = i
                break

    cmd =["weka.classifiers.trees.RandomForest","-p",str(class_index),"-l",str(model_file),"-T",str(arff_file)]
    (stdout, stderr)  = java(cmd, classpath=weka_class_path,stdout=subprocess.PIPE,stderr=subprocess.PIPE)
    err_msg = stderr.decode("GBK")
    if err_msg !="":
        raise OSError('Java command failed : ' + str(err_msg))
    result= stdout.decode(stdin.encoding)
    if "prediction ()" not in result:
        return None
    result = result[result.index("prediction ()")+"prediction ()".__len__():].strip()
    tmp = result.split("\n")
    final_result ={}
    for t in tmp:
        result_tmp = re.split(" +", t.strip())
        if result_tmp[0] not in final_result.keys():
            predict = result_tmp[2].split(":")[1]
            final_result[result_tmp[0]]= [predict,float(result_tmp[-1])]
    print(final_result)
    return final_result


weka_classify("../res/arff_data/train/大西洋.csv.arff","../res/model/大西洋/NaiveBayes.model")