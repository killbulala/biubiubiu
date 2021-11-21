# -*- coding: utf-8 -*-
"""
@Time ： 2021/11/21 14:08
@Auth ： killbulala
@File ：voc2txt.py
@IDE ：PyCharm
@Email：killbulala@163.com
voc2txt
"""
import os
import xml.etree.ElementTree as ET


def voc2txt(r, voc_dict, f):
    ann_root = os.path.join(r, 'VOC2012', 'Annotations')
    img_root = os.path.join(r, 'VOC2012', 'JPEGImages')
    xml_lst = os.listdir(ann_root)
    for xml_file in xml_lst:
        flag_obj = False
        flag_wh = False
        flag_exists = False
        doc = ''
        root = ET.parse(os.path.join(ann_root, xml_file)).getroot()
        filename = root.find('filename').text
        if os.path.exists(os.path.join(img_root, filename)):
            flag_exists = True
        doc += os.path.join(img_root, filename) + ' '
        objs = root.findall('object')
        if len(objs) > 0:
            flag_obj = True
        for obj in objs:
            name = obj.find('name').text
            bnd = obj.find('bndbox')
            xmin = bnd.find('xmin').text
            doc += xmin + ','
            ymin = bnd.find('ymin').text
            doc += ymin + ','
            xmax = bnd.find('xmax').text
            doc += xmax + ','
            ymax = bnd.find('ymax').text
            doc += ymax + ','
            cls_id = voc_dict[name]
            doc += cls_id + ' '
            w = int(float(xmax)) - int(float(xmin))
            h = int(float(ymax)) - int(float(ymin))
            if w > 0 and h > 0:
                flag_wh = True
        if flag_exists and flag_obj and flag_wh:
            f.write(doc.strip() + '\n')

def get_voc_dict(txt):
    dict = {}
    with open(txt, 'r') as f:
        lines = f.readlines()
        for i, line in enumerate(lines):
            dict[line.strip()] = str(i)
    return dict


if __name__ == '__main__':
    root = r'F:\killbulala\work\datasets\VOCdevkit'
    with open(os.path.join(root, 'killbulala_voc.txt'), 'w') as f:
        voc_dict = get_voc_dict('voc_classes.txt')
        voc2txt(root, voc_dict, f)




