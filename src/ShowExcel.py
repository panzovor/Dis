import src.preprocess as feature


def show_excel(filename,gap):
    filepath = "../res/名字与文章可能出现歧义的公司/"+filename
    result,name = feature.read_csv(filepath)
    for var in result:
        print(len(result),var)
        input()
    # no = int(input().strip())
    # while no>=0:
    #     tmp = result[max(0,no-gap):min(no+gap+1,len(result))]
    #     if len(tmp) == 1:
    #         print(tmp)
    #     else:
    #         for var in tmp:
    #             print(var)
    #         no = int(input().strip())

filename = "东方财富.csv"
show_excel(filename,0)