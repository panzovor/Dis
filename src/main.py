import src.features as features
import src.analyzeresult as analyzer
import os
import src.trainer as trainer
import src.preprocess as preprocess

def prepare_data():
    ### 从原始数据中提取出相应的句子
    # preprocess.preprocess()

    ## 将标注的文件切分成 训练数据和测试数据
    features.seperate(new_only=True)
    print("seperate_done")

    ## 对训练数据进行特征提取统计
    features.train(new_only=True)
    print("trian done")

    ## 将训练数据和测试数据进行特征化
    features.featurelize(new_only =True)
    print("featurelize done")

    features.arff(new_only=True)
    print("arff done")

def train_predict(train_file,test_file,model_save_root,result_save_root):
    # print(model_save_root)
    if not os.path.exists(model_save_root):
        # if os.path.isdir(model_save_root):
        os.mkdir(model_save_root)
        print("make "+model_save_root)
    trainer.train_model(train_file,model_save_root)
    # trainer.test_model(model_save_root,test_file,result_save_root)
    # return model_save_root

def disambugation():
    train_root = "../res/arff_data/train/"
    test_root = "../res/arff_data/test/"
    model_save_root ="../res/model/"
    result_save_root ="../res/result/"

    if not os.path.exists(model_save_root):
        os.makedirs(model_save_root)
    if not os.path.exists(result_save_root):
        os.makedirs(result_save_root)

    for name in os.listdir(train_root):
        train_predict(train_root+name,test_root+name,model_save_root+name[:-9],result_save_root)
        # analysze_res_root(result_save_root)

def analyze_result():
    ## 测试前，需使用weka进行训练，得到最后的分类结果
    ## 对每个模型进行测试

    analyzer.newest_date = "20170803"

    # model_names = ["RandomForest","MultilayerPercptron","SMO","RBF","BayesNet","NaiveBayes"]
    # for line in model_names:
    #     print(line)


    model_files = "../res/model/test_details/" + analyzer.newest_date + "/"
    analysze_res_root(model_files)

def analysze_res_root(model_files):
    model_names = os.listdir(model_files)
    print('\n'.join(model_names)+"\n")
    tmp = []
    for name in model_names:
        tmp.append(analyzer.analyze(model_files + name))
    print()
    for i in range(len(model_names)):
        name = model_names[i]
        res = analyzer.get_info(model_files + name)
        res = [name] + res + tmp[i]
        print('\t'.join(map(str, res)))


if __name__ =="__main__":

    preprocess.combine()
    prepare_data()
    disambugation()
   # analyze_result()


