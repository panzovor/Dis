
import os
import src.tools as tools
import src.preprocess as preprocess
import sys
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer
import networkx as nx
import datetime



## 用于计算词性及关键词特征的类
class characteristic_feature():

    def __init__(self):
        parameter_file = "../res/parameter/characteristic/keywords.txt"
        tmp = tools.read_txt(parameter_file)
        # print(tmp)
        self.keywords = {}
        self.features_num = 6

        for t in tmp:
            if t[0] not in self.keywords.keys():
                self.keywords[t[0]] = []
            self.keywords[t[0]].append(t[1:])



    ## input:
    ##     tag: 句子的分词的词性结果
    ##     labels: 对应的词性类别
    ## output:
    ##     res: 词性类别对应的次数（如[1,2,3,4]: 该句子有1个名词，2个动词，3个形容词，4个数词）
    ## label = n : 名词， v:动词，a: 形容词， m:数词
    def count_character(self,tag,labels= ["n","v","a","m"]):
        res = [0,0,0,0]
        for var in tag:
            for i in range(len(labels)):
                if labels[i] in var:
                    res[i]+=1
        return res

    ## input:
    ##     key: 目标词（如东方财富等）
    ##     sentence: 待分类句子
    ## output:
    ##     res: 各个类别的次数（如[0,1]: 0 类： 出现0词，1类：出现1次）
    def count_keywords(self,key,sentence):
        res = [0,0]
        if key in self.keywords.keys():
            for tmp in self.keywords[key]:
                # print(tmp)
                if tmp[0] in sentence:
                    res[int(tmp[-1])]+=1
        if "all" in self.keywords.keys():
            for tmp in self.keywords["all"]:
                if tmp[0] in sentence:
                    res[int(tmp[-1])] +=1
        return  res


    ## input:
    ##     key: 目标词（如东方财富等）
    ##     sentence: 待分类句子
    ## output:
    ##      chara+keywor
    ##          chara:词性统计结果
    ##          keywor: 关键词统计结果
    ##      示例： noun_count,verb_count,adj_count,numeral_count,neg_class_count,pos_class_count
    def get_features(self,key,sentence):
        words,tag = tools.seperate_sentence(sentence,tag=True)
        chara = [var/len(words) for var in self.count_character(tag)]
        keywor = self.count_keywords(key,sentence)
        return chara+keywor

## abanoned用于计算tfidf值特征的类
class tfidf_feature():

    def __init__(self):
        self.parameter_file = "../res/parameter/tfidf.csv"
        self.tfidf_map =tools.read_into_dict(self.parameter_file)
        self.tfidf_num = 3

    ## input:
    ##     key: 目标词（如东方财富等）
    ##     sentence: 待分类句子
    ##     seperate_words: 是否进行分词处理
    ## output:
    ##     tfidf0+tfidf1
    ##         (0类中tfidf 值最大的前self.tfidf_num个词的tfidf值，不足补0
    ##          1类中tfidf 值最大的前self.tfidf_num个词的tfidf值，不足补0)
    ##      示例：tfidf0_word_value1,tfidf0_word_value2,tfidf0_word_value3, tfidf1_word_value1,tfidf1_word_value2,tfidf1_word_value3
    def get_features(self,key,sentence,seperate_words = True):
        tfidf0,tfidf1 = [],[]
        if key in self.tfidf_map.keys():
            if seperate_words:
                words,tag = tools.seperate_sentence(sentence)
                tmp0,tmp1= [],[]
                for word in words:
                    if word in self.tfidf_map[key].keys():
                        tmp0.append(float(self.tfidf_map[key][word][0]))
                        tmp1.append(float(self.tfidf_map[key][word][1]))
                tmp0.sort(reverse=True)
                tmp1.sort(reverse=True)
                while len(tmp0)<self.tfidf_num:
                    tmp0.append(0.0)
                while len(tmp1)<self.tfidf_num:
                    tmp1.append(0.0)
                tfidf0 = tmp0[:self.tfidf_num]
                tfidf1 = tmp1[:self.tfidf_num]
            else:
                tmp0,tmp1 = [],[]
                for word in self.tfidf_map[key].keys():
                    if word in sentence:
                        tmp0.append(float(self.tfidf_map[key][word][0]))
                        tmp1.append(float(self.tfidf_map[key][word][1]))
                tmp0.sort(reverse=True)
                tmp1.sort(reverse=True)
                while len(tmp0) < self.tfidf_num:
                    tmp0.append(0.0)
                while len(tmp1) < self.tfidf_num:
                    tmp1.append(0.0)
                tfidf0 = tmp0[:self.tfidf_num]
                tfidf1 = tmp1[:self.tfidf_num]
        return tfidf0+tfidf1

    def tfidf(self,corpus,keylabelmap,positive = True):
        countermatrix = CountVectorizer(token_pattern=r"(?u)\b\w+\b")
        tfidfmatrix = TfidfTransformer()
        tf = countermatrix.fit_transform(corpus)
        tfidfv = tfidfmatrix.fit_transform(tf)
        tfidfv = tfidfv.toarray()
        res ={}
        whole_words=  countermatrix.get_feature_names()
        for index in keylabelmap.keys():
            key,label = keylabelmap[index]
            if key not in res.keys():
                res[key] = {}
            if label not in res[key].keys():
                res[key][label] ={}
            for i in range(len(whole_words)):
                if positive and tfidfv[index][i] == 0:
                    continue
                if whole_words[i] not in res[key][label].keys():
                    res[key][label][whole_words[i]] = 0
                res[key][label][whole_words[i]] =tfidfv[index][i]

        fres = {}
        for key in res.keys():
            if key not in fres.keys():
                fres[key] = {}
            for label in res[key].keys():
                for w in res[key][label].keys():
                    if w not in fres[key].keys():
                        fres[key][w] = [0]* len(res[key].keys())
                    fres[key][w][int(label)] = res[key][label][w]

        return fres


    ### csv_data : keyword , sentences ,..., label
    ### return :   keyword->label ->words
    def generate_label_doc(self,csv_data):
        feature_data ={}
        for line in csv_data:
            if line[0] not in feature_data.keys():
                feature_data[line[0]] = {"0":[],"1":[]}
            string  = str(max(0,min(1, int(line[-1]))))
            feature_data[line[0]][string].append(line[1])
        res = {}
        for key in feature_data.keys():
            if key not in res.keys():
                res[key] = {}
            for label in feature_data[key].keys():
                if label not in res[key].keys():
                    res[key][label] = ""
                for sen in feature_data[key][label]:
                    words,tag = tools.seperate_sentence(sen)
                    res[key][label]+= ' '.join(words)+' '
        return res


    ### csv_data: keyword->label ->words
    ### return :  keyword->words->[label0_tfidf, label1_tfidf]
    def calculate_tfidf(self,csv_data):
        featuredata = self.generate_label_doc(csv_data)
        corpus =[]
        key_label_map = {}
        for key in featuredata.keys():
            for label in featuredata[key].keys():
                key_label_map[len(corpus)] = [key,label]
                corpus.append(featuredata[key][label])
        tfidfv = self.tfidf(corpus,key_label_map)

        return tfidfv


    def train(self,fileroot,save_file= None,rate = 1):
        filelist = os.listdir(fileroot)
        filelist = [fileroot+var for var in filelist]
        csv_data = tools.read_multi_csv(filelist)
        if rate> 0 and rate<1:
            csv_data,test = preprocess.seperate_data(csv_data,rate)
            # print(len(csv_data),len(test),len(csv_data))
        self.tfidf_map = self.calculate_tfidf(csv_data)
        if save_file != None:
            self.parameter_file = save_file
        tools.save_tfidf_csv(self.tfidf_map,self.parameter_file)

### abandoned
class probability_feature():

    def __init__(self):
        self.parameter_label_file = "../res/parameter/probability_label.txt"
        self.parameter_words_file = "../res/parameter/probability_words.txt"
        self.parameter_wall_file = "../res/parameter/probability_wall.txt"
        self.pro_num = 3
        self.parameter_label={}
        self.parameter_words={}
        self.parameter_wall = {}

        # self.load_parameter()


    ## 计算训练文件中各个词的正负分布
    ## 在关键词（keywords：如东方财富等）中每个词的正负类比例
    ## 在关键词（keywords：如东方财富等）中该关键词的正负类比例
    ## 在词语（word：如股票，购买，出售等）中每个词的正负类比例
    ## 三个结果分别存于： words_file,label_file,wall_file,
    ## input: 训练文件的文件夹
    ## output: res,key_map,all
    ##        (self.parameter_words) : key-> words-> [label_0_pro,label_1_pro]
    ##        (self.parameter_label): key->[label_0_pro,label_1_pro]
    ##        (self.parameter_wall) all : word->[label_0_pro,label_1_pro]
    def train(self,train_root):
        filelist = os.listdir(train_root)
        filelist = [train_root + var for var in filelist]
        csv_data = tools.read_multi_csv(filelist)
        ### csv_data : keyword , sentences ,..., label
        self.parameter_words = {}
        self.parameter_label = {}
        i = 0
        for line in csv_data:

            if line[0] not in self.parameter_words.keys():
                self.parameter_words[line[0]] = {}
            words,tag = tools.seperate_sentence(line[1])
            i+=1
            # print(i,len(csv_data))

            lab = max(0,min(1, int(line[-1])))
            if line[0] not in self.parameter_label.keys():
                self.parameter_label[line[0]] = [0.0,0.0]
            self.parameter_label[line[0]][lab]+=1

            # print(line[0],self.parameter_words[line[0]])

            for word in words:
                if word not in self.parameter_words[line[0]].keys():
                    self.parameter_words[line[0]][word] = [0.0,0.0]
                self.parameter_words[line[0]][word][lab]+=1

        # print("words,label done")

        key_map_save = []
        for key in self.parameter_label.keys():
            tmp = self.parameter_label[key]
            tmp_v = sum(tmp)
            if tmp_v>0:
                self.parameter_label[key] =[var/tmp_v for var in tmp]
            key_map_save.append([key] + self.parameter_label[key])

        res_save = []
        self.parameter_wall= {}
        for key in self.parameter_words.keys():
            for word in self.parameter_words[key].keys():
                if word not in self.parameter_wall.keys():
                    self.parameter_wall[word] = [0.0,0.0]
                self.parameter_wall[word][0]+= self.parameter_words[key][word][0]
                self.parameter_wall[word][1] += self.parameter_words[key][word][1]
                tmp = self.parameter_words[key][word]
                tmp_v = sum(tmp)
                if tmp_v>0:
                    self.parameter_words[key][word] =[var/tmp_v for var in tmp]
                res_save.append([key,word]+self.parameter_words[key][word])

        # print("parameter_wall done")

        all_save= []
        for word in self.parameter_wall.keys():
            tmp = self.parameter_wall[word]
            tmp_v = sum(self.parameter_wall[word])
            if tmp_v> 0:
                self.parameter_wall[word] = [var/tmp_v for var in tmp]
            all_save.append([word]+self.parameter_wall[word])

        tools.save_txt(key_map_save,self.parameter_label_file)
        tools.save_txt(res_save,self.parameter_words_file)
        tools.save_txt(all_save,self.parameter_wall_file)

        return self.parameter_words,self.parameter_label,self.parameter_wall

    ## 加载参数文件
    def load_parameter(self):
        tmp = tools.read(filepath=self.parameter_label_file).strip().split("\n")
        for line in tmp:
            line = line.split(",")
            self.parameter_label[line[0]] = [float(var) for var in line[1:]]

        tmp = tools.read(filepath=self.parameter_words_file).strip().split("\n")
        for line in tmp:
            line = line.split(",")
            if line[0] not in self.parameter_words.keys():
                self.parameter_words[line[0]]={}

            # try:
            if line[1] not in self.parameter_words[line[0]].keys():
                self.parameter_words[line[0]][line[1]]= [float(var) for var in line[2:]]
            # except:
            #     print(line[0])
            #     print(line[1])
            #     input()

        tmp = tools.read(filepath=self.parameter_wall_file).strip().split("\n")
        for line in tmp:
            line = line.split(",")
            if line[0] not in self.parameter_wall.keys():
                self.parameter_wall[line[0]] = []
            self.parameter_wall[line[0]] = [float(var) for var in line[1:]]

    ## input:
    ##     key: 关键词（例如东方财富等）
    ##     sentence:待分类句子
    ## output:
    ##     key_words_value + label_value + words_value
    ##     key_words_value: 负类中概率最高的前self.pro_num个词语的概率值，正类中概率最高的前self.pro_num个词语的概率值
    ##     label_value : 该关键词的歧义概率
    ##     words_value: 句子中出现的， 负类词总体样本概率最高的前self.pro_num个词语的概率值，正类词总体样本概率最高的前self.pro_num个词语的概率值
    def get_features(self,key,sentence):
        keywords_value,label_value,words_value = [],[],[]
        words,tag = tools.seperate_sentence(sentence)
        if len(self.parameter_words) >0:
            tmp0,tmp1 = [],[]
            if key in self.parameter_words.keys():
                for word in words:
                    if word in self.parameter_words[key].keys():
                        tmp0.append(self.parameter_words[key][word][0])
                        tmp1.append(self.parameter_words[key][word][1])
                tmp0.sort()
                tmp1.sort()
                while len(tmp0) <self.pro_num:
                    tmp0.append(0.0)
                while len(tmp1) <self.pro_num:
                    tmp1.append(0.0)
                keywords_value = tmp0[:self.pro_num]+tmp1[:self.pro_num]

        if len(self.parameter_label)>0:
            if key in self.parameter_label.keys():
                label_value = self.parameter_label[key]

        if len(self.parameter_wall)>0:
            tmp0,tmp1 = [],[]
            for word in words:
                if word in self.parameter_wall.keys():
                    tmp0.append(self.parameter_wall[word][0])
                    tmp1.append(self.parameter_wall[word][1])

            tmp0.sort()
            tmp1.sort()
            while len(tmp0) <self.pro_num:
                tmp0.append(0.0)
            while len(tmp1) < self.pro_num:
                tmp1.append(0.0)
            words_value = tmp0[:self.pro_num]+ tmp1[:self.pro_num]
        return keywords_value+label_value
        # return keywords_value+label_value+words_value

class single_class_probability_feature():

    def __init__(self):
        self.parameter = {}
        self.sour_para = {}
        self.features_size = 4
        self.parameter_filepath = "../res/parameter/probability/"
        self.load_parameter()
        self.features_num = self.features_size*2
        self.new_only = False

    def train(self,key_words,filepath):
        train_data = tools.read_csv_format(filepath,collum=6)
        # train_data = tools.read(filepath).strip().split("\n")
        root = self.parameter_filepath+key_words+"/"
        if not os.path.exists(root):
            os.mkdir(root)
        para_file = root+"pro.txt"
        sour_file = root+"sour.txt"
        if self.new_only:
            if os.path.exists(para_file) and os.path.exists(sour_file):
                return None


        if key_words not in self.parameter.keys():
            self.parameter[key_words] = {}
            self.sour_para[key_words] = {}

        for line in train_data:
            # tmp = line.split(",")
            words,tag = tools.seperate_sentence(line[1].strip())
            for w in words:
                if w not in self.parameter[key_words].keys():
                    self.parameter[key_words][w] =[0.0,0.0,0.0,0.0]
                if int(line[-1]) >0 :
                    self.parameter[key_words][w][1] +=1
                else:
                    self.parameter[key_words][w][0] +=1
            if line[3] not in self.sour_para[key_words].keys():
                self.sour_para[key_words][line[3]] = [0.0,0.0,0.0,0.0]
            if int(line[-1])>0:
                self.sour_para[key_words][line[3]][1]+=1
            else:
                self.sour_para[key_words][line[3]][0]+=1

        para_content =[]
        sour_content = []
        for word in self.parameter[key_words].keys():
            value0 = self.parameter[key_words][word][0]
            value1 = self.parameter[key_words][word][1]
            self.parameter[key_words][word][2] = value0/(value0+value1)
            self.parameter[key_words][word][3] = value1/(value0+value1)
            para_content.append([word]+self.parameter[key_words][word])
        for sour in self.sour_para[key_words].keys():
            # for word in self.sour_para[key_words][sour].keys():
            value0 = self.sour_para[key_words][sour][0]
            value1 = self.sour_para[key_words][sour][1]
            self.sour_para[key_words][sour][2] = value0 / (value0 + value1)
            self.sour_para[key_words][sour][3] = value1 / (value0 + value1)
            sour_content.append([sour]+self.sour_para[key_words][sour])

        tools.save_txt(para_content,para_file)
        tools.save_txt(sour_content,sour_file)

    def load_parameter(self):
        filelist = os.listdir(self.parameter_filepath)
        for name in filelist:
            path_pro = self.parameter_filepath+name+"/pro.txt"
            path_sour = self.parameter_filepath+name+"/sour.txt"
            if name not in self.parameter.keys():
                self.parameter[name] = {}
                self.sour_para[name] = {}
            content = tools.read(path_pro).strip().split("\n")
            for line in content:
                line= line.split(",")
                # try:
                self.parameter[name][line[0]] = [float(var) for var in line[1:]]
                # except:
                #     print(name,line)
                #     input()
            content = tools.read(path_sour).strip().split("\n")
            for line in content:
                line = line.split(",")
                self.sour_para[name][line[0]] = [float(var) for var in line[1:]]

    def get_features(self,key_words,sentence):
        # print(key_words,self.parameter.keys())
        if key_words in self.parameter.keys():
            words,tag = tools.seperate_sentence(sentence)
            pro_pos =[]
            pro_neg =[]
            for w in words:
                if w in self.parameter[key_words].keys():
                    pro_neg.append(self.parameter[key_words][w][2])
                    pro_pos.append(self.parameter[key_words][w][3])

            while len(pro_pos)< self.features_size:
                pro_pos.append(0.0)
                pro_neg.append(0.0)
            pro_pos.sort(reverse=True)
            pro_neg.sort(reverse=True)
            return pro_pos[:self.features_size]+pro_neg[:self.features_size]

        else:
            return None

class texrank():
    def __init__(self):
        self.rank_file  = "../res/parameter/textrank/"
        self.feature_size = 4
        self.ranking_result = {}
        self.load_parameter()
        self.min_count = 2
        self.features_num = self.feature_size*2
        self.new_only = False

    def calculate_rank_score(self,data):
        words = []
        line_inwords = []
        for line in data:
            word,tag = tools.seperate_sentence(line)
            line_inwords.append(word)
            for w in word:
                if w not in words:
                    words.append(w)
        words_matrix = [[0.0] * len(words) for var in words]
        for line in line_inwords:
            for i in range(len(line) - 1):
                for j in range(i + 1, len(line)):
                    wi = words.index(line[i])
                    wj = words.index(line[j])
                    words_matrix[wi][wj] += 1
                    words_matrix[wj][wi] += 1
        edges = []
        print(len(words_matrix))
        for i in range(len(words_matrix)):
            for j in range(len(words_matrix)):
                if i != j and words_matrix[i][j]>self.min_count:
                    edges.append((i,j,words_matrix[i][j]))
                    edges.append((j,i,words_matrix[j][i]))
        print(len(edges))
        g = nx.DiGraph()
        g.add_weighted_edges_from(edges)
        res = nx.pagerank_numpy(g)
        result = {}
        for index, value in res.items():
            result[words[index]] = value
        return result

    ### single label data
    def train(self,key_words,data_file):
        if key_words not in self.ranking_result.keys():
            self.ranking_result[key_words]={}


        path_pos = self.rank_file+key_words+"/rank_pos.txt"
        path_neg = self.rank_file+key_words+"/rank_neg.txt"

        if os.path.exists(path_pos) and os.path.exists(path_neg):
            return None

        data = tools.read_csv_format(data_file,collum=6)
        pos_data = []
        neg_data = []
        for line in data:
            if int(line[-1])>0 :
                pos_data.append(line[1])
            else:
                neg_data.append(line[1])

        pos_score = self.calculate_rank_score(pos_data)
        neg_score = self.calculate_rank_score(neg_data)
        self.ranking_result[key_words]["pos"] = pos_score
        self.ranking_result[key_words]["neg"] = neg_score

        pos_content = []
        for w, v in pos_score.items():
            pos_content.append([w, v])
        neg_content = []
        for w, v in neg_score.items():
            neg_content.append([w, v])


        tools.save_txt(pos_content,filepath=path_pos)
        tools.save_txt(neg_content,filepath=path_neg)

    def load_parameter(self):
        def load(rank_file,key_words,filename):
            content = tools.read(rank_file + key_words + "/"+filename).strip().split("\n")
            res = {}
            for line in content:
                tmp = line.split(",")
                if len(tmp) == 2:
                    res[tmp[0]] = float(tmp[1])
                else:
                    print("text rank feature loading failure: uncorrect format", line)
                    input()
            return res

        filelist=  os.listdir(self.rank_file)
        for key_words in filelist:
            self.ranking_result[key_words]["pos"] = load(self.rank_file,key_words,"rank_pos.txt")
            self.ranking_result[key_words]["neg"] = load(self.rank_file,key_words,"rank_neg.txt")



    def get_features(self,key_words,sentences):
        words,tag = tools.seperate_sentence(sentences)
        pos_v = []
        neg_v = []
        for w in words:
            if w in self.ranking_result[key_words]["pos"].keys():
                pos_v.append(self.ranking_result[key_words]["pos"][w])
            if w in self.ranking_result[key_words]["neg"].keys():
                neg_v.append(self.ranking_result[key_words]["neg"][w])
        while len(pos_v)< self.feature_size:
            pos_v.append(0.0)
        while len(pos_v)< self.feature_size:
            pos_v.append(0.0)
        pos_v.sort()
        neg_v.sort()
        return pos_v+neg_v



character = characteristic_feature()
# tfidf = tfidf_feature()
textrank = texrank()
probability = single_class_probability_feature()

parameter = {"probability":True,"textrank":False}

def get_features(key,sentences):

    cfeature = character.get_features(key,sentences)
    # tfidffeature = tfidf.get_features(key,sentences,seperate_words=seperate_words)
    res = cfeature
    if parameter["textrank"]:
        textrankfeature = textrank.get_features(key,sentences)
        res= res+textrankfeature
    if parameter["probability"]:
        profeature = probability.get_features(key,sentences)
        res = res+ profeature
    return res

root = "../res/labeldata/"
# probability.train(train_root=root)
# print("probability done")
# tfidf.train(root)
# print("tfidf done")

# print(len(probability.parameter_words))
# print(len(probability.parameter_label))
# print(len(probability.parameter_wall))

def featurelize_data(root,save_root,new_only = False):
    filelist= os.listdir(root)
    features_num = character.features_num
    if parameter["probability"]:
        features_num+= probability.features_num
    if parameter["textrank"]:
        features_num+= textrank.features_num



    j=0
    title = []
    for v in range(features_num):
        title.append("att"+str(v))
    title.append("class")

    for file in filelist:
        if new_only and os.path.exists(save_root+file):
            continue
        features = []
        features.append(title)
        j+=1
        i = 0
        filepath = root+file
        data = tools.read_csv_format(filepath,collum=6)
        print(file)
        # patch = int(len(data) / 10) if int(len(data))/10 > 0 else 1
        for line in data:
            i += 1
            # if i% patch == 0:
            #     sys.stdout.write("*")
            #     sys.stdout.flush()
            tmp = [line[0]+"_"+line[2]]
            tmp.extend(get_features(line[0],line[1]))
            tmp.append(str(max(0,min(1,int(line[-1])))))
            features.append(tmp)
        # print()
        tools.save_csv(features,save_root+file)

def demo():
    label = "名词，动词，形容词，数词，关键词，tfidf0_0，tfidf0_1，tfidf0_2，tfidf1_0，tfidf1_1，tfidf1_2"
    print(character.keywords)
    keyword = "东方财富"
    test_sen1 = "两会调查之民生篇：住房话题最火爆 房产税被认是猛药 (东方财富网)"
    test_sen2 = "东方财富股票具有较大的升值空间，可以入手"
    test_sen3 = "东方财富具有较大的升值空间，可以入手"
    feat1 = get_features(keyword, test_sen1)
    feat2 = get_features(keyword, test_sen2)
    feat3 = get_features(keyword, test_sen3)

    print(label)
    print(test_sen1)
    print(feat1)

    print(test_sen2)
    print(feat2)
    print(test_sen3)
    print(feat3)

def train(new_only =False):
    probability.new_only = new_only
    textrank.new_only = new_only

    trainroot = "../res/train_labeldata/"
    filelist = os.listdir(trainroot)
    for name in filelist:
        print(name[:-4], trainroot + name)
        if parameter["probability"]:
            probability.train(name[:-4],trainroot+name)
            print("probability done")
        if parameter["textrank"] :
            textrank.train(name[:-4],trainroot+name)
            print("textrank done")

def featurelize(new_only = False):
    trainroot = "../res/train_labeldata/"
    testroot = "../res/test_labeldata/"

    save_train_root = "../res/feature_data/train/"
    save_test_root = "../res/feature_data/test/"

    if not os.path.exists(save_train_root):
        os.makedirs(save_train_root)

    if not os.path.exists(save_test_root):
        os.makedirs(save_test_root)

    featurelize_data(trainroot, save_train_root,new_only)
    featurelize_data(testroot, save_test_root,new_only)

def arff(new_only=False):
    save_train_arff_root = "../res/arff_data/train/"
    save_test_arff_root = "../res/arff_data/test/"

    save_train_root = "../res/feature_data/train/"
    save_test_root = "../res/feature_data/test/"

    if not os.path.exists(save_train_arff_root):
        os.makedirs(save_train_arff_root)
    if not os.path.exists(save_test_arff_root):
        os.makedirs(save_test_arff_root)

    preprocess.csv2arff_all(save_train_root,save_train_arff_root,new_only)
    preprocess.csv2arff_all(save_test_root,save_test_arff_root,new_only)

def seperate(new_only = True):
    root = "../res/labeldata/"
    trainroot = "../res/train_labeldata/"
    testroot = "../res/test_labeldata/"
    if not os.path.exists(trainroot):
        os.mkdir(trainroot)
        print("make "+trainroot)
    if not os.path.exists(testroot):
        os.mkdir(testroot)
        print("make "+testroot)
    if not new_only:
        preprocess.seperate_data(root,trainroot,testroot,rate=0.75)
    else:
        preprocess.seperate_new_data(root,trainroot,testroot,rate = 0.75)

if  __name__ == "__main__":

    # seperate()
    # print("seperate_done")
    # train()
    # print("trian done")
    featurelize()

