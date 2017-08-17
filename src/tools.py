import sys
import jieba
import csv
import jieba.posseg as pog
import os
import xlrd
import chardet

xlrd.Book.encoding="utf-8"
encoding = "GB2312"
csv.field_size_limit(2147483647)
def read_csv(filepath):
    res = []
    name = filepath[filepath.rindex("/")+1:-4]
    while name[-1].isdigit():
        name= name[:-1]
    with open(filepath,mode="r",errors="ignore") as file:
        csvfile =  csv.reader(file)
        # try:
        for row in csvfile:
            # if name in row[-1]:
            res.append(row)
    return res,name

def create_dir_if_not_exist(filepath):
    root = filepath[:filepath.rindex("/")+1]
    # print("dd",root)
    if not os.path.exists(root):
        # print(root)
        os.makedirs(root)

def read_csv_format(filepath,collum = 0):
    res =[]
    with open(filepath,mode="r",errors="ignore") as file:
        csvfile =  csv.reader(file)
        i =0
        for row in csvfile:
            if i == 0:
                i=1
                continue
            if collum<=0 or len(row) == collum:
                res.append(row)
    return res

def open_xls(filepath):
    data = xlrd.open_workbook(filepath)
    return data

def read_xls_solidcollum(filepath,collum,skip_first = True,sheet_no = 0):
    if collum>0:
        res = []
        data = open_xls(filepath)
        table = data.sheets()[sheet_no]
        row_num = table.nrows
        start_row = 1 if skip_first else 0
        for i in range(start_row,row_num):
            row = table.row_values(i)
            # for var in row:
                # print(var,var.encode("utf-8"))
                # print(var,var.encode("gbk"))
                # print(var,var.encode("gb2312"))
                # input()
            res.append([var for var in  row[:collum]])

        return res
    else:
        return None



def read_multi_csv(filelist):
    res = []
    for var in filelist:
        print(var)
        tmp = read_csv_format(var,6)
        res.extend(tmp)
    print(len(res))
    return res

def read(filepath,skip_fisrt=False):
    res = []
    with open(filepath) as file:
        if skip_fisrt:
            i=0
            for line in file:
                if i== 0:
                    i+=1
                    continue
                res.append(line.split(","))
        else:
            for line in file:
                res.append(line.split(","))
    return res

def read(filepath,encoding = "utf-8"):

    with open(filepath,encoding=encoding,errors="ignore") as file:
        return file.read()

def read_txt(filepath,collum_num = 3,encoding = "utf-8",seperate = ","):
    res = []
    with open(filepath,encoding=encoding,mode="r") as file:
        for line in file:
            if seperate in line:
                t = line.split(seperate)
                if t[0] == "key":
                    continue
                if collum_num>0 and len(t) == collum_num:
                    res.append(t)
    return res


def read_into_dict(filepath):
    res = {}
    with open(filepath, mode="r", errors="ignore") as file:
        csvfile = csv.reader(file)
        i = 0
        for row in csvfile:
            if i == 0:
                i = 1
                continue
            if len(row) == 4:
                if row[0] not in res.keys():
                    res[row[0]] = {}
                if row[1] not in res[row[0]].keys():
                    res[row[0]][row[1]] =[]
                res[row[0]][row[1]] = row[2:]
    return res

def save_sen_csv(data,filepath):
    # seperator = "$#$"
    with open(filepath,mode="w",newline='') as file:
        csvwriter = csv.writer(file)
        csvwriter.writerow(["keywords","sentence","article_no","source","title or content"])
        for key in data.keys():
            for line in data[key]:
                csvwriter.writerow([key]+line)

def save_csv(data,filepath):
    with open(filepath, mode="w", newline='',errors='ignore') as file:
        csvwriter = csv.writer(file)
        for line in data:
            csvwriter.writerow(line)

def delete(path):
    if os.path.exists(path):
        os.remove(path)

def save_txt(content,filepath,encoding = "utf-8"):
    with open(filepath,mode="w",encoding=encoding) as file:
        listtype = True
        if isinstance(content[0],str):
            listtype = False
        for line in content:
            if listtype:
                line = list(map(str,line))
                file.write(','.join(line)+"\n")
            else:
                file.write(line)

def save_tfidf_csv(data,filepath):
    # seperator = "$#$"
    with open(filepath,mode="w",newline='') as f:
        file = csv.writer(f)
        file.writerow(["name","words","tfidf_0","tfidf_1"])

        for key in data.keys():
            for word in data[key].keys():
                tmp = [key,word]
                tmp.extend(data[key][word])
                # if key == "北京城乡" and word == "0":
                #     print(tmp)
                file.writerow(tmp)


user_dict = "../res/parameter/segmentation/user_dict.txt"
jieba.load_userdict(user_dict)
stop_words_file = "../res/parameter/segmentation/stop_words.txt"
stop_words =[]
with open(stop_words_file, encoding="utf-8") as file:
    for line in file:
        stop_words.append(line.strip())

        ## input:
        ##     sentence: 待分类句子
        ## output:
        ##     words: 分词结果
        ##     tag: 分词结果中每个词对应的词性

'''
### input: 待分词句子
### output: words:分词结果
###         tag   分词结果中每个词语对应的词性
'''
def seperate_sentence( sentence,tag = False):
    sentence = sentence.strip()
    sentence = sentence.replace("\"","")
    sentence = sentence.replace("“","")
    sentence = sentence.replace("”","")
    sentence = sentence.replace("\",\"","")
    sentence = sentence.replace("\n","")
    if tag:
        tmp = pog.cut(sentence)
        words, tag = [], []
        for var in tmp:
            if var.word in stop_words:
                continue
            words.append(var.word)
            tag.append(var.flag)
        return words, tag
    else:
        return [var for var in  list(jieba.cut(sentence)) if var not in stop_words],None


if __name__ == "__main__":
    import networkx as nx
    import time
    # edges = [("我","哎",10),
    #          ("北京","天安门",5),
    #          ("北京","故宫",7),
    #          ("北京","颐和园",4),
    #          ("其他","等等",2)]
    # g = nx.DiGraph()
    # g.add_weighted_edges_from(edges)
    # start = time.time()
    # score = nx.pagerank_numpy(g)
    # end = time.time()
    # print(end-start)
    # for key in score.keys():
    #     print(key,score[key])
    import datetime

    # print(str(datetime.date.today()).replace("-","_"))