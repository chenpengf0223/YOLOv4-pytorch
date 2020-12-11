import sys

sys.path.append(".")
import xml.etree.ElementTree as ET
import config.yolov4_config as cfg
import os
from tqdm import tqdm
import io
import json

def parse_custom_annotation(data_path, anno_path, data_list_path):
    """
    phase pascal voc annotation, eg:[image_global_path xmin,ymin,xmax,ymax,cls_id]
    :param data_path: eg: VOC\VOCtrainval-2007\VOCdevkit\VOC2007
    :param anno_path: path to ann file
    :return: batch size of data set
    """
    if cfg.TRAIN["DATA_TYPE"] == "Customer":
        classes = cfg.Customer_DATA["CLASSES"]
    else:
        print('not custom data.')
        input()
    image_num = 0
    folder_list = os.listdir(data_path)

    data_list_f = open(data_list_path, 'w')
    with open(anno_path, "w") as f:
        for folder in folder_list:
            class_id = classes.index(folder)
            f_list = os.listdir(data_path+'/'+folder)
            for file_tp in f_list:
                file_cur = data_path+'/'+folder+'/'+file_tp
                filename, file_type = os.path.splitext(file_cur)
                if file_type == '.json':
                    print('file_cur', file_cur)
                    with io.open(file_cur, 'r', encoding='utf-8') as fe:#gbk
                        anno = json.load(fe)
                    obj_list = anno['outputs']['object']
                    img_w = anno['size']['width']
                    img_h = anno['size']['height']
                    img_c = anno['size']['depth']
                    new_str = ''
                    for obj in obj_list:
                        if obj['name'] != folder:
                            print('')
                            input()
                        box = obj['bndbox']
                        x_min = box['xmin']
                        x_max = box['xmax']
                        y_min = box['ymin']
                        y_max = box['ymax']
                        if (x_min<0 or x_min>=img_w or y_min<0 or y_min>=img_h or 
                            x_max<0 or x_max>=img_w or y_max<0 or y_max>=img_h):
                            print('box is error')
                            input()
                        new_str += " " + ",".join(
                            [str(x_min), str(y_min), str(x_max), str(y_max), str(class_id)]
                            )

                    if new_str == '':
                        continue
                    image_num += 1
                    image_path = filename + '.jpg'
                    annotation = image_path
                    annotation += new_str
                    annotation += "\n"
                    # print(annotation)
                    f.write(annotation)
                    
                    data_list_f.write(file_tp[:-5] + '\n')
    data_list_f.close()
    return image_num


if __name__ == "__main__":
    train_data_path_2007 = '/home/chenp/YOLOv4-pytorch/qixing'
    train_annotation_path = "/home/chenp/YOLOv4-pytorch/data/train_annotation.txt"
    if os.path.exists(train_annotation_path):
        print('remove train annotation path...')
        input()
        os.remove(train_annotation_path)
    train_data_list_path = '/home/chenp/YOLOv4-pytorch/data/VOCtest-2007/VOCdevkit/VOC2007/ImageSets/Main/train.txt'

    test_data_path_2007 = '/home/chenp/YOLOv4-pytorch/qixing'
    test_annotation_path = "/home/chenp/YOLOv4-pytorch/data/test_annotation.txt"
    if os.path.exists(test_annotation_path):
        print('remove test annotation path...')
        input()
        os.remove(test_annotation_path)
    test_data_list_path = '/home/chenp/YOLOv4-pytorch/data/VOCtest-2007/VOCdevkit/VOC2007/ImageSets/Main/test.txt'

    len_train = parse_custom_annotation(
        train_data_path_2007,
        train_annotation_path,
        train_data_list_path
    )
    len_test = parse_custom_annotation(
        test_data_path_2007,
        test_annotation_path,
        test_data_list_path
    )

    print(
        "The number of images for train and test are :train : {0} | test : {1}".format(
            len_train, len_test
        )
    )