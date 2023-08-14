# -*- coding: utf-8 -*-
# @Author  : ssbuild
# @Time    : 2023/8/14 8:55
import json
from pprint import pprint



if __name__ == '__main__':

    filename = r'E:\py-http\ToolBench\data_example\answer\G1_answer\57_ChatGPT_DFS_woFilter_w2.json'
    with open(filename,mode='r',encoding='utf-8') as f:
        jd = json.loads(f.read())

    print(jd)

    pprint(jd)