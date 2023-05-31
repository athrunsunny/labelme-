#  -*- coding:utf-8 -*-
# Author:Mr.屌丝
import json
import os.path
from hashlib import sha1
import requests
from fake_useragent import UserAgent


class BaiduSpider:

    def __init__(self):
        # https://img02.sogoucdn.com/app/a/100520020/ebdc830ef3ecedcea306c02cba235320
        # https://image.baidu.com/search/albumsdata?pn=30&rn=30&tn=albumsdetail&word=%E5%AE%A0%E7%89%A9%E5%9B%BE%E7%89%87&album_tab=%E5%8A%A8%E7%89%A9&album_id=687&ic=0&curPageNum=1
        # 初始化一下 头部 把刚刚变动的数据改成大括号 便于后面传参
        self.url = 'https://image.baidu.com/search/albumsdata?pn={}&rn=30&tn=albumsdetail&word=%E5%AE%A0%E7%89%A9%E5%9B%BE%E7%89%87&album_tab=%E5%8A%A8%E7%89%A9&album_id=703&ic=0&curPageNum={}'
        self.headers = {
            'User-Agent': UserAgent().random
        }

    def sha1(self, href):
        # 进行sha1的加密
        s = sha1()
        s.update(href.encode())
        return s.hexdigest()

    def parse_html(self, url):
        img_html = requests.get(url=url, headers=self.headers).text
        # 我们这里需要把数据转换成json格式的数据
        img_json = json.loads(img_html)
        # print(img_json)
        # print(len(img_json['albumdata']['linkData']))

        for href in img_json['albumdata']['linkData']:
            img_href = href['thumbnailUrl']
            # 调用下载图片的函数 需要把图片的链接传参过去
            self.img_info(img_href)

    def img_info(self, href):
        # 二级页面请求 下载图片
        html = requests.get(url=href, headers=self.headers).content
        # 进行一个sha1的加密
        sha1_r = self.sha1(href)
        # 创建一个保存图片的路径
        file = 'F:\Study\Spider\img\\'
        if not os.path.exists(file):
            os.makedirs(file)
        # 完整保存图片的链接
        filename = file + sha1_r + '.jpg'
        # 判断有没有这个保存图片的路径  没有则创建
        if not os.path.exists(file):
            os.mkdir(file)
        # 进行图片保存
        with open(filename, 'wb') as f:
            f.write(html)
            print(filename)

    def crawl(self):
        # 这里进行计算是应该我们看到有185个图片 一个请求只有30个 所以我们计算一下需要发送几个请求
        # page = 185 // 30 if 185 % 30 == 0 else 185 // 30 + 1
        cn = 99
        page = cn // 30 if cn % 30 == 0 else cn // 30 + 1
        for number in range(page):
            pn = number * 30
            self.parse_html(self.url.format(pn, number))


if __name__ == '__main__':
    baidu = BaiduSpider()
    baidu.crawl()




# import requests
# import json
# import urllib
#
# def getSogouImag(category,length,path):
#     n = length
#     cate = category
#     imgs = requests.get('http://pic.sogou.com/pics/channel/getAllRecomPicByTag.jsp?category='+cate+'&tag=%E5%85%A8%E9%83%A8&start=0&len='+str(n))
#     jd = json.loads(imgs.text)
#     jd = jd['all_items']
#     imgs_url = []
#     for j in jd:
#         imgs_url.append(j['bthumbUrl'])
#     m = 0
#     for img_url in imgs_url:
#             print('***** '+str(m)+'.jpg *****'+'   Downloading...')
#             urllib.request.urlretrieve(img_url,path+str(m)+'.jpg')
#             m = m + 1
#     print('Download complete!')
#
# getSogouImag('壁纸',2000,'d:/download/壁纸/')
