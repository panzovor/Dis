import src.tools as tools

jingdu = 3

newest_date = "20170803"

def get_data(result_file):

    result = tools.read(result_file)
    start_label = "probability distribution"
    end_label = "Summary"
    result = result[result.rindex(start_label) + len(start_label): result.rindex(end_label)].strip()
    tmp = result.split("\n")
    res = []
    for t in tmp:
        t = t.strip().split()
        if len(t)>=5:
            res.append([int(t[0]), int(t[1][-1]), int(t[2][-1]), float(t[-2] if "*" not in t[-2] else t[-2][1:])])
    return  res

def get_info(result_file):
    # result_file = "../res/model/test_details/" +newest_date+"/"+ modelname + ".txt"
    result = tools.read(result_file)
    prefix ="classified as"
    subfix = "b = 1"

    content = result[result.rindex(prefix)+len(prefix):result.rindex(subfix)].strip()
    content = content.split("\n")
    content1 = content[0][:content[0].index("|")].strip().split()
    content2 = content[1][:content[1].index("|")].strip().split()
    matrix = [content2[1],content2[0],content1[0],content1[1]]
    matrix = list(map(int,matrix))
    matrix.append(sum(matrix))
    res = calculate(matrix)
    res = matrix+res
    # print("\t".join(map(str,res)))
    return res


### input: 混淆矩阵
### output: 正类准确率，正类召回率，正类F值，负类准确率，负类召回率，负类F值，样本正类比例，负类排除率，误排率，分类准确率
def calculate(matrix):
    p_precision = matrix[0]/(matrix[0]+matrix[3]) if matrix[0]+matrix[3]>0 else 1
    p_recall = matrix[0] /(matrix[0]+matrix[1]) if matrix[0]+matrix[1] >0 else 1

    n_precision = matrix[2]/(matrix[2]+matrix[1]) if matrix[2]+matrix[1] >0 else 1
    n_recall = matrix[2]/(matrix[2]+matrix[3])  if matrix[2]+matrix[3]>0 else 1

    paichu = n_recall
    paiwu = 1-p_recall
    zhunq = p_precision
    f_score_p = 2*p_precision*p_recall/(p_precision+p_recall) if (p_precision+p_recall)>0 else 0
    f_score_n = 2*n_precision*n_recall/(n_precision+n_recall) if (n_precision+n_recall)>0 else 0
    precision_unfilter = round(sum(matrix[:2])/sum(matrix[:4]),jingdu)
    return [round(p_precision,jingdu),round(p_recall,jingdu),round(f_score_p,jingdu),round(n_precision,jingdu),round(n_recall,jingdu),round(f_score_n,jingdu),
            round(paichu,jingdu),round(paiwu,jingdu),round(zhunq,jingdu),precision_unfilter]

def analyze(result_file,test_file = "../res/train_featuredata/test.csv"):
    csv_data = tools.read_csv_format(test_file)
    result_data = get_data(result_file)
    analyze_res(csv_data,result_data)

def analyze_res(csv_data,result_data):
    # result_file = "../res/model/test_details/"+newest_date+"/"+modelname+".txt"
    # csv_data = tools.read_csv_format(test_file)
    # result_data =get_data(result_file)

    ## key->no->[[real_label1,pre_label1],...,[real_labeln,pre_labeln]]
    data ={}
    for i in range(len(result_data)):
        result_line = result_data[i]
        test_line = csv_data[i]
        # data.append([test_line[0],result_line[0],test_line[-1]]+result_line[1:])
        key = test_line[0].split("_")
        if key[0] not in data.keys():
            data[key[0]] = {}
        if key[1] not in data[key[0]].keys():
            data[key[0]][key[1]] =[]
        data[key[0]][key[1]].append([int(test_line[-1]),int(result_line[2])])

    res = {}


    for name in data.keys():
        if name  not in res.keys():
            ## 正类判准，正类判误，负类判准，负类判误，样本总数
            res[name] = [0,0,0,0,0]
        for no in data[name].keys():
            real = int(max( [ var[0] for var in data[name][no]]))
            pred = int(max([var[1] for var in data[name][no]]))
            res[name][-1]+=1
            if real == pred :
                if real == 1:
                    res[name][0]+=1
                else:
                    res[name][2]+=1
            else:
                if real == 1:
                    res[name][1]+=1
                else:
                    res[name][3]+=1

    result =[]
    for name in res.keys():
        cres = calculate(res[name])
        result.append([res[name][-1]]+res[name][:4]+cres)
        print('\t'.join(map(str,[name]+res[name]+cres)))
    value =[0.0,0.0,0.0,0.0]
    count = 0
    pos,all =0,0
    for line in result:
        count+= line[0]
        value[0]+= line[0]*line[-4]
        value[1]+= line[0]*line[-3]
        value[2]+= line[0]*line[-2]
        # print(line[1:3],line[1:5])
        pos+= sum(line[1:3])
        all += sum(line[1:5])

    value[0] = round(value[0]/count,jingdu)
    value[1] = round(value[1]/count,jingdu)
    value[2] = round(value[2]/count,jingdu)
    value[3] = round(pos / all, jingdu)

    # print('\t'.join(map(str,value)))
    print()
    return value




model_names =[]
# analyze("random_forest")
# analyze("MultilayerPercptron")
# analyze("SMO")
# analyze("RBF")
# analyze("BayesNet")
# analyze("NaiveBayes")

# get_info("random_forest")
# get_info("MultilayerPercptron")
# get_info("SMO")
# get_info("RBF")
# get_info("BayesNet")
# get_info("NaiveBayes")