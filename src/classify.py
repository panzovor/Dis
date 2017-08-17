import src.preprocess as pre
import src.train_py as trainer
import src.tools as tools
import src.features as feature

model_root = "../res/model/"
parameter ="../res/parameter/result.csv"
models ={}
def load_models():
    content = tools.read_csv_format(parameter)
    for line in content:
        if line[0] not in models.keys():
            model_path = "../res/model/"+line[0]+"/"+line[1]+".model"
            models[line[0]] = trainer.load_model(model_path)

def classify(text):
    sents = pre.extract_sentence_from_text(text)
    result  ={}
    for key in models.keys():
        for sen in sents:
            if key in sen:
                features_key = feature.get_features(key,sen)
                predict = models[key].predict(features_key)
                if str(predict[0]) == "1":
                    if sen not in result.keys():
                        result[sen] = []
                    result[sen].append(key)

    feature_all = feature.get_features("all",sen)
    predict = models["all"].predict(feature_all)
    if str(predict[0]) == "1":
        result[sen].append("all")

    # for sen in result:
    #     print(sen,result[sen])
    return result

if __name__ == "__main__":
    content = "中国石油集团成功发行首单100亿元可交换公司债券"
    load_models()
    res = classify(content)
    print(len(res))