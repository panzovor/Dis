import requests
from lxml import etree

cookie = {"Cookie": "#YOUR_COOKIE"}
user_id =0


def get_page_num():
    url = 'http://weibo.cn/u/%d?filter=1&page=1' % user_id
    html = requests.get(url, cookies=cookie).content
    print(u'user_id和cookie读入成功')
    selector = etree.HTML(html)
    pageNum = (int)(selector.xpath('//input[@name="mp"]')[0].attrib['value'])
    return pageNum

def get_single_page(page):
    url = 'http://weibo.cn/u/%d?filter=1&page=%d' % (user_id, page)
    lxml = requests.get(url, cookies=cookie).content
    # 文字爬取
    selector = etree.HTML(lxml)
    content = selector.xpath('//span[@class="ctt"]')
    return content

def save_weibo(content):
    path = "D://czb//weibo//"+str(user_id)
    with open(path,encoding="utf-8",mode="a") as file:
        file.write(content)

def get_weibo(start,end):
    for page in range(start,end):
        con = get_single_page(page)
        save_weibo(con)


