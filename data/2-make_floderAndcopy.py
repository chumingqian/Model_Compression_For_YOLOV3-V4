import xml.etree.ElementTree as ET
import pickle
import os
from os import listdir, getcwd
from os.path import join
import shutil

"""
sets=[('2012', 'train'), ('2012', 'val'), ('2007', 'train'), ('2007', 'val'), ('2007', 'test')]

classes = ["aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat", "chair", "cow", "diningtable", "dog", "horse", "motorbike", "person", "pottedplant", "sheep", "sofa", "train", "tvmonitor"]

# XML坐标格式转换成yolo坐标格式 
def convert(size, box):
    dw = 1./(size[0])
    dh = 1./(size[1])
    x = (box[0] + box[1])/2.0 - 1
    y = (box[2] + box[3])/2.0 - 1
    w = box[1] - box[0]
    h = box[3] - box[2]
    x = x*dw
    w = w*dw
    y = y*dh
    h = h*dh
    return (x,y,w,h)

# 标记文件格式转换
def convert_annotation(year, image_id):
    in_file = open('VOCdevkit/VOC%s/Annotations/%s.xml'%(year, image_id))
    out_file = open('VOCdevkit/VOC%s/labels/%s.txt'%(year, image_id), 'w')
    tree=ET.parse(in_file)
    root = tree.getroot()
    size = root.find('size')
    w = int(size.find('width').text)
    h = int(size.find('height').text)

    for obj in root.iter('object'):
        difficult = obj.find('difficult').text
        cls = obj.find('name').text
        if cls not in classes or int(difficult)==1:
            continue
        cls_id = classes.index(cls)
        xmlbox = obj.find('bndbox')
        b = (float(xmlbox.find('xmin').text), float(xmlbox.find('xmax').text), float(xmlbox.find('ymin').text), float(xmlbox.find('ymax').text))
        bb = convert((w,h), b)
        out_file.write(str(cls_id) + " " + " ".join([str(a) for a in bb]) + '\n')

wd = getcwd()

for year, image_set in sets:
    if not os.path.exists('VOCdevkit/VOC%s/labels/'%(year)):
        os.makedirs('VOCdevkit/VOC%s/labels/'%(year))
    image_ids = open('VOCdevkit/VOC%s/ImageSets/Main/%s.txt'%(year, image_set)).read().strip().split()
    list_file = open('%s_%s.txt'%(year, image_set), 'w')
    for image_id in image_ids:
        list_file.write('%s/VOCdevkit/VOC%s/JPEGImages/%s.jpg\n'%(wd, year, image_id))
        convert_annotation(year, image_id)
    list_file.close()
"""

"""
# 合并多个文本文件
def mergeTxt(file_list,outfile):
    with open(outfile,'w') as wfd:
        for f in file_list:
            with open(f,'r') as fd:
                shutil.copyfileobj(fd, wfd)

file_list= ['2007_train.txt', '2007_val.txt', '2012_train.txt', '2012_val.txt']
outfile = 'train.txt'
mergeTxt(file_list, outfile)
file_list= ['2007_train.txt', '2007_val.txt', '2007_test.txt', '2012_train.txt', '2012_val.txt']
outfile = 'train.all.txt'
mergeTxt(file_list, outfile)

"""


# 创建VOC文件夹及子文件夹
wd = os.getcwd()
data_base_dir = os.path.join(wd, "VOC/")
if not os.path.isdir(data_base_dir):
    os.mkdir(data_base_dir)
img_dir = os.path.join(data_base_dir, "images/")
if not os.path.isdir(img_dir):
    os.mkdir(img_dir)    
img_train_dir = os.path.join(img_dir, "train/")
if not os.path.isdir(img_train_dir):
    os.mkdir(img_train_dir)
img_val_dir = os.path.join(img_dir, "val/")
if not os.path.isdir(img_val_dir):
    os.mkdir(img_val_dir)
label_dir = os.path.join(data_base_dir, "labels/")
if not os.path.isdir(label_dir):
    os.mkdir(label_dir)    
label_train_dir = os.path.join(label_dir, "train/")
if not os.path.isdir(label_train_dir):
    os.mkdir(label_train_dir)
label_val_dir = os.path.join(label_dir, "val/")
if not os.path.isdir(label_val_dir):
    os.mkdir(label_val_dir)

print(os.path.exists('train.txt'))
f = open('train.txt', 'r')
lines = f.readlines()

# 使用train.txt中的图片作为yolov5的训练集
for line in lines:
    line = line.replace('\n', '')
    if (os.path.exists(line)):
        shutil.copy(line, "VOC/images/train") # 复制图片
        print('copying train img file  %s' % line + '\n')
        
    line = line.replace('JPEGImages', 'labels') # 复制label
    line = line.replace('jpg', 'txt')
    if (os.path.exists(line)):
        shutil.copy(line, "VOC/labels/train")
        print('copying train label file  %s' % line + '\n')

# 使用2007_test.txt中的图片作为yolov5的验证集
print(os.path.exists('2007_test.txt'))
f = open('2007_test.txt', 'r')
lines = f.readlines()

for line in lines:
    line = line.replace('\n', '')
    if (os.path.exists(line)):
        shutil.copy(line, "VOC/images/val") # 复制图片
        print('copying val img file  %s' % line + '\n')
        
    line = line.replace('JPEGImages', 'labels')
    line = line.replace('jpg', 'txt')
    if (os.path.exists(line)):
        shutil.copy(line, "VOC/labels/val") # 复制label
        print('copying val img label %s' % line + '\n')
