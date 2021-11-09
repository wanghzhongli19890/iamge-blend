import os
import cv2 as cv
import xml.etree.ElementTree as ET



def xml_jpg2labelled(imgs_path, xmls_path, labelled_path):
    imgs_list = os.listdir(imgs_path)
    xmls_list = os.listdir(xmls_path)
    nums = len(imgs_list)
    for i in range(200):
        imgs_list1 = imgs_list[i].split('.')
        img_path = os.path.join(imgs_path, imgs_list[i])
        xml_path = os.path.join(xmls_path, imgs_list1[0]+'.xml')
        img = cv.imread(img_path)
        labelled = img
        root = ET.parse(xml_path).getroot()
        objects = root.findall('object')
        for obj in objects:
            name = obj.find('name').text
            print(name)
            bbox = obj.find('bndbox')
            xmin = int(float(bbox.find('xmin').text.strip()))
            ymin = int(float(bbox.find('ymin').text.strip()))
            xmax = int(float(bbox.find('xmax').text.strip()))
            ymax = int(float(bbox.find('ymax').text.strip()))
            labelled = cv.rectangle(labelled, (xmin, ymin), (xmax, ymax), (0, 0, 255), 3)
            labelled = cv.putText(labelled, name,(xmin, ymin), cv.FONT_HERSHEY_COMPLEX, 2.0, (255, 250, 0), 2)
            print(imgs_list[i])
        cv.imwrite(labelled_path+'/'+'%s_labelled.jpg' % (imgs_list[i].split('.')[0]), labelled)
        # cv.imshow('labelled', labelled)
        # cv.imshow('origin', origin)
        # cv.waitKey()


if __name__ == '__main__':
    imgs_path = '/home/wzl/code/paper-code/paper-affordance-pose/affrodance-paper/data-generation/copy-paste-summary/datasets/JPEGImages'
    xmls_path = '/home/wzl/code/paper-code/paper-affordance-pose/affrodance-paper/data-generation/copy-paste-summary/datasets/Annotations'
    labelled_path = '/home/wzl/code/paper-code/paper-affordance-pose/affrodance-paper/data-generation/copy-paste-summary/datasets/jpglabel'
    xml_jpg2labelled(imgs_path, xmls_path, labelled_path)
