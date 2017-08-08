import os
import src.preprocess as features

feature_len = 1000


#name	label	words	tfidf
def load_feature(filepath):
    csv_data = features.read_csv_format(filepath,4)
    res = {}
    for line in csv_data:
        if line[0] not in res.keys():
            res[line[0]] = {}
        if line[1] not in res[line[0]].keys():
            res[line[0]][line[1]] =[]
        if words_filter(line[1]):
            continue
        res[line[0]][line[1]] = [float(var) for var in line[2:]]
    return res

def words_filter(features):
    if len(features) >10:
        return False


tmp = load_feature("../res/features_result/tfidf.csv")
for key in tmp.keys():
    for w in tmp[key].keys():
        print(key,w,tmp[key][w])

def train(train_data):
    pass


def test(test_data):
    pass