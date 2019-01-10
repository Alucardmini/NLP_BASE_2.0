# -*- coding:utf-8 -*-
# @version: 1.0
# @author: wuxikun
# @date: '1/6/19'

import os
import sys

curPath = os.path.abspath(os.path.dirname(__file__))
rootPath = os.path.split(curPath)[0]
sys.path.append(rootPath)
sys.path.append(os.path.split(os.path.abspath(os.path.dirname(os.path.dirname(__file__))))[0])

# sys_path_str = 'os.path.split(os.path.abspath(os.path.dirname(__file__)))[0]'
#                # 'sys.path.append(rootPath)\n' \
#                # 'sys.path.append(os.path.split(os.path.abspath(os.path.dirname(os.path.dirname(__file__))))[0])'
#
# rootPath = eval(sys_path_str)
# print(rootPath)



'''
将文件下中的所有文件存成csv格式
csv格式: label_index, label_name, content(content为input_folder下的一个文件正文)

'''
def product_csv_file_from_src_file(input_folder, output_path, label_index, label_name):
    file_names = os.listdir(input_folder)  # 当前文件夹下所有文件
    csv_content = []
    for file_name in file_names:
        with open(os.path.join(input_folder, file_name), 'r') as f:
            content = [line.strip().replace('\n', '').replace(',', '') for line in f.readlines()]
            content = ''.join(content)
        csv_content.append(','.join([label_index, label_name, content]))

    # csv_title = 'label_index, custom_label, content \n'
    with open(output_path, 'a+')as f:
        # f.write(csv_title)
        f.writelines('\n'.join(csv_content))

if __name__ == "__main__":
    label_list = ["工程类别", "电梯", "监理", "设计"]
    data_path = rootPath + '/data'
    output_path = rootPath + '/train_data'
    for index, _ in enumerate(label_list):
        product_csv_file_from_src_file(os.path.join(data_path, _), os.path.join(output_path, _+'.csv'), label_index='__label__' + str(index+1), label_name=_)


