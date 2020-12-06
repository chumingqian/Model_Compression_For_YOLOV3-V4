# -*- coding: utf-8 -*-
"""
需要修改的地方：
1. sets中替换为自己的数据集
3. 将本文件放到VOC2007目录下
4. 直接开始运行
"""

import xml.etree.ElementTree as ET
import pickle
import os
from os import listdir
from os.path import join

#sets = [('0712', 'train'), ('0712', 'val'), ('0712', 'test')]  #替换为自己的数据集
# 确保images 文件中有 train, val 文件夹； 并且最终生成的txt 文件是以以下集合命名的；
sets = [('0712', 'train'), ('0712', 'val')]  
                                                    


def gen_final_txt(sets):
    imgfilepath = 'VOC/images'

    for year, image_set in sets:
        current_path = imgfilepath +'/'+ image_set
        current_img = os.listdir(current_path)
        current_num = len(current_img)
        list = range(current_num)
        
        list_file = open('%s_%s.txt' % (year, image_set),
                         'w')
        for i in list:
           name = current_img[i][:-4] 
           print(name)
           list_file.write('data/VOC/images/%s/%s.jpg\n' % (image_set,name) )
        list_file.close()


"""
        image_ids = open('%s.txt' %
                         (image_set)).readlines()  #.strip()#.split()
        # print(image_ids)
        print('*' * 20)
        list_file = open('%s_%s.txt' % (year, image_set),
                         'w')
        for image_id in image_ids:
            image_id = image_id[:-1]
            print(image_id)
            list_file.write('data/images/%s0712/%s.jpg\n' % (image_set ,image_id))        
        list_file.close()
"""
if __name__ == "__main__":
   gen_final_txt(sets)
