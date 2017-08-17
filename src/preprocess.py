import jieba
import src.tools as tools
import os
import re


sentence_regrex = "。|！|？"

def extract_sentence_from_text(text):
    sents = re.split(sentence_regrex,text)
    return sents

## map:
##      key:   keywords, sentences, label
##      value: col index
def extract_sentence(csvdata, keywords, sentence = [0,3],source = 1):
    res = {}
    for i in range(len(csvdata)):
        line = csvdata[i]
        if keywords not in res.keys():
            res[keywords] = []
        sents = re.split(sentence_regrex,line[sentence[0]])
        for sent in sents:

            if keywords in sent:
                if "," in sent:
                    sent = sent.replace(",", "，")
                if "\n" in sent:
                    sent = sent.replace("\n","")
                if "\r\n" in sent:
                    sent = sent.replace("\r\n","")
                if len(sent)>200:
                    sent=  sent[max(0,sent.index(keywords)-100):min(sent.index(keywords)+100,len(sent))]
                if len(line)<6:
                    res[keywords].append([sent,i,line[source],"c"])
                else:
                    res[keywords].append([sent,i,line[source],"c",line[-1]])
        if keywords in line[sentence[1]]:
            sent = line[sentence[1]]
            if "," in sent:
                sent = sent.replace(",", "，")
            if len(line)<6:
                res[keywords].append([sent,i,line[source],"t"])
            else:
                res[keywords].append([sent,i,line[source],"t",line[-1]])

    return res


mid_root  = "../res/名字与文章可能出现歧义的公司_去燥/"

def transfer_data(csvfile,save_file):
    csv_data1,name = tools.read_csv(csvfile)
    # print(len(csv_data1))
    tools.save_csv(csv_data1,mid_root+save_file[save_file.rindex("/")+1:])
    sentences = extract_sentence(csv_data1,name)
    # #保存未标注语料
    tools.save_sen_csv(sentences, save_file)
    return len(csv_data1)


def preprocess(root  ="../res/名字与文章可能出现歧义的公司/",saveroot="../res/labeldata/"):
    filelist = os.listdir(root)
    res = []
    for filename in filelist:
        res.append(filename)
    print(res)
    fail_res = []
    succ_res = []
    for name in res:
        print(name,end="")
        l = transfer_data(root+name,saveroot+name)
        succ_res.append(name)
        print(" done",l)
    print("fail key words",fail_res)
    print(succ_res)
    return res

def seperate_input_data(data,rate,two_class=True,min_size = 20):
    train_data ,test_data =[],[]
    data_dict = {}
    for line in data:
        tmp = line[0] +line[-1]
        if two_class:

            line[-1] = "0" if  int(line[-1])<0 else line[-1]
            tmp = line[0] +str(min(1,int(line[-1])))
        if tmp not in data_dict.keys():
            data_dict[tmp] = []
        data_dict[tmp].append(line)

    for key in data_dict.keys():
        data_now = data_dict[key]
        train_size = min(max(int(len(data_now)*rate),min_size),len(data_now))

        test_size = min(max(int(len(data_now)*(1-rate)),min_size),len(data_now))

        # print(key,train_size,test_size,len(data_now))

        train_data.extend(data_now[:train_size])
        test_data.extend(data_now[-test_size:])
    print(len(data),len(train_data),len(test_data))
    return train_data,test_data

def combine():
    root ="../res/labeldata/"
    filelist = os.listdir(root)
    res = {}
    for var in filelist:
        if var[-5].isdigit():
            if var[:-5] not in res.keys():
                res[var[:-5]] =[]
            res[var[:-5]].append(var)
        else:
            if var[:-4] not in res.keys():
                res[var[:-4]]=[]
            res[var[:-4]].append(var)

    for key in res.keys():
        if len(res[key]) == 1:
            continue
        else:
            content = tools.read(root+res[key][0],encoding="GBK")
            if content[-1] !="\n":
                content+="\n"
            for i in range(1,len(res[key])):
                tmp = tools.read(root+res[key][i],encoding="GBK")
                tmp = tmp.split("\n",maxsplit=2)[-1]
                content+= tmp
                if content[-1] !="\n":
                    content+="\n"
            tools.save_txt(content,root+key+res[key][0][-4:],encoding="GBK")
            for name in res[key]:
                tools.delete(root+name)

def combine_new(fileroot,save_root):
    filelist_news = os.listdir(fileroot+"媒体新闻/")
    filelist_weixin = os.listdir(fileroot+"微信/")
    news_data = {}
    for name in filelist_news:
        filename = fileroot+"媒体新闻/"+name
        csv_data = tools.read_xls_solidcollum(filename,collum=3,sheet_no=0)
        cleanname = name[:name.index(".")]
        cleanname = cleanname.replace("微信", "")
        cleanname = cleanname.replace("新闻", "")
        for line in csv_data:
            if cleanname not in news_data.keys():
                news_data[cleanname] =[]
            news_data[cleanname].append([line[2],"媒体新闻",line[0],line[1],cleanname,1])

        csv_data = tools.read_xls_solidcollum(filename, collum=3, sheet_no=1)
        cleanname = name[:name.index(".")]
        cleanname = cleanname.replace("微信", "")
        cleanname = cleanname.replace("新闻", "")
        for line in csv_data:
            if cleanname not in news_data.keys():
                news_data[cleanname] = []
            news_data[cleanname].append([line[2], "媒体新闻", line[0], line[1], cleanname,0])

    for name in filelist_weixin:
        filename = fileroot + "微信/" + name
        csv_data = tools.read_xls_solidcollum(filename, collum=3,sheet_no=0)
        cleanname = name[:name.index(".")]
        cleanname = cleanname.replace("微信","")
        cleanname = cleanname.replace("新闻","")
        for line in csv_data:
            if cleanname not in news_data.keys():
                news_data[cleanname] = []
            news_data[cleanname].append([line[2], "微信",line[0], line[1], cleanname,1])

        csv_data = tools.read_xls_solidcollum(filename, collum=3, sheet_no=1)
        cleanname = name[:name.index(".")]
        cleanname = cleanname.replace("微信", "")
        cleanname = cleanname.replace("新闻", "")
        for line in csv_data:
            if cleanname not in news_data.keys():
                news_data[cleanname] = []
            news_data[cleanname].append([line[2], "微信", line[0], line[1], cleanname, 0])
    for cleanname in news_data.keys():
        print(cleanname, len(news_data[cleanname]))

    for key in news_data.keys():
        content = news_data[key]
        tools.save_csv(content,filepath=save_root+key+".csv")

def seperate_data(root,train_root,test_root,rate , two_class = True,min_size = 20):
    filelist = os.listdir(root)
    for file in filelist:
        filepath = root+file
        data = tools.read_csv_format(filepath,collum=6)
        train,test = seperate_input_data(data,rate,two_class,min_size)
        train_save = train_root+file
        test_save = test_root+file
        tools.save_csv(train,train_save)
        tools.save_csv(test,test_save)
        print(file,"done")

def seperate_new_data(root,train_root,test_root,rate,two_class= True,min_size = 20):
    filelist = os.listdir(root)
    train_filelist = os.listdir(train_root)
    test_filelist = os.listdir(test_root)
    train_test_filelist = set(train_filelist).intersection(set(test_filelist))
    for file in filelist:
        if file in train_test_filelist:
            continue
        filepath = root+file
        data = tools.read_csv_format(filepath,collum=6)
        train,test = seperate_input_data(data,rate,two_class,min_size)
        train_save = train_root+file
        test_save = test_root+file
        tools.save_csv(train,train_save)
        tools.save_csv(test,test_save)
        print(file,"done")

def csv2arff_all(csv_root,arff_root,new_only= False):
    csvs = os.listdir(csv_root)
    for name in csvs:
        csv_file = csv_root+name
        arff_file = arff_root+name+".arff"
        if new_only and os.path.exists(arff_file):
            continue
        csv2arff(csv_file,arff_file)

def csv2arff(csv_file,arff_file):
    csv_data = tools.read_csv_format(csv_file)
    attribute_nums = len(csv_data[0][1:-1])
    title = "@relation tmp\n"
    for i in range(attribute_nums):
        title+= "@attribute att"+str(i)+" numeric\n"
    title+= "@attribute class {0,1}\n@data\n"

    content = ""
    for var in csv_data:
        content+=','.join(var[1:])+"\n"

    content = title+content

    tools.save_txt(content,arff_file)

def combine_all_label_files(root= "../res/labeldata/",save_path = "../res/labeldata/all.csv"):
    filelist = os.listdir(root)
    all_content = [["keywords","sentence","article_no","source","title or content"]]
    for name in filelist:
        if "all" not in name:
            content = tools.read_csv_format(root+name)
            for line in content:
                if line[0] == "keywords":
                    continue
                all_content.append(["all"]+line[1:])
    tools.save_csv(all_content,save_path)





if  __name__ == "__main__":
    names = ["刘志成","乐远","潘开迪"]
    for n in names:
        root = "../res/new_data/"+n+"/"
        save  ="../res/名字与文章可能出现歧义的公司/"
        combine_new(root,save)
    # preprocess(root,save)