import src.features as features
import os
import src.train_py as trainer
import src.analyzeresult as analyresult
import src.preprocess as preprocess

only_new = True

def perpare_data2label():
    ## 从原始数据中提取出相应的句子
    preprocess.preprocess()
    print("labeled done")

def prepare_data():

    ## 将标注的文件切分成 训练数据和测试数据
    features.seperate(new_only=only_new)
    print("seperate_done")

    ## 对训练数据进行特征提取统计
    features.train(new_only=only_new)
    print("trian done")

    ## 将训练数据和测试数据进行特征化
    features.featurelize(new_only =only_new)
    print("featurelize done")

    # features.arff(new_only=True)
    # print("arff done")            name = name[:name.index(".")]


def disambugation():
    train_root = "../res/feature_data/train/"
    test_root = "../res/feature_data/test/"
    model_save_root = "../res/model/"
    filelist = os.listdir(train_root)
    models = trainer.models
    fail = []
    for name in filelist:
        train_file = train_root+name
        test_file = test_root+name
        if "." in name:
            name = name[:name.index(".")]
        # print("training: using trainfile( ",train_file,"),testfile( ",test_file,")")

        print(name)
        for modelname in models.keys():
            # print(modelname)
            try:
                trainer.train_model(train_file,test_file,model_name=modelname,model_save_path=model_save_root+name+"/"+modelname+".model",res_save_path=model_save_root+name+"/"+modelname+".txt")
                # print("      saving model in path:(", model_save_root + name + "/" + modelname + ".model", " )")
                # print("      saving result in path:( ", model_save_root + name + "/" + modelname + ".txt", " )")
            except Exception as e:
                print(str(e))
                fail.append([name,modelname])
    for var in fail:
        print(var)

def analyze_result():
    data = analyresult.load_all_result()
    res = analyresult.select_best(data)

if __name__ =="__main__":

    # preprocess.combine_new()

    # names = ["刘志成", "乐远", "潘开迪"]
    # for n in names:
    #     root = "../res/new_data/" + n + "/"
    #     save = "../res/名字与文章可能出现歧义的公司/"
    #     preprocess.combine_new(root, save)

    # perpare_data2label()
    # prepare_data()
    disambugation()
    analyze_result()
    # preprocess.combine_all_label_files()

