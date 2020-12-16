from xml.dom.minidom import Document
import os
import io
import os.path
from PIL import Image
import json

imgs_path = "/home/chenp/YOLOv4-pytorch/qixing-data/test"

def writeXml(tmp, imgname, w, h, objbud, wxml):
    doc = Document()
    #owner
    annotation = doc.createElement('annotation')
    doc.appendChild(annotation)
    #owner
    folder = doc.createElement('folder')
    annotation.appendChild(folder)
    folder_txt = doc.createTextNode("VOC2005")
    folder.appendChild(folder_txt)
 
    filename = doc.createElement('filename')
    annotation.appendChild(filename)
    filename_txt = doc.createTextNode(imgname)
    filename.appendChild(filename_txt)
    #ones#
    source = doc.createElement('source')
    annotation.appendChild(source)
 
    database = doc.createElement('database')
    source.appendChild(database)
    database_txt = doc.createTextNode("The VOC2005 Database")
    database.appendChild(database_txt)
 
    annotation_new = doc.createElement('annotation')
    source.appendChild(annotation_new)
    annotation_new_txt = doc.createTextNode("PASCAL VOC2005")
    annotation_new.appendChild(annotation_new_txt)
 
    image = doc.createElement('image')
    source.appendChild(image)
    image_txt = doc.createTextNode("flickr")
    image.appendChild(image_txt)
    #onee#
    #twos#
    size = doc.createElement('size')
    annotation.appendChild(size)
 
    width = doc.createElement('width')
    size.appendChild(width)
    width_txt = doc.createTextNode(str(w))
    width.appendChild(width_txt)
 
    height = doc.createElement('height')
    size.appendChild(height)
    height_txt = doc.createTextNode(str(h))
    height.appendChild(height_txt)
 
    depth = doc.createElement('depth')
    size.appendChild(depth)
    depth_txt = doc.createTextNode("3")
    depth.appendChild(depth_txt)
    #twoe#
    segmented = doc.createElement('segmented')
    annotation.appendChild(segmented)
    segmented_txt = doc.createTextNode("0")
    segmented.appendChild(segmented_txt)
 
    for i in range(0, len(objbud)//5):
        #threes#
        object_new = doc.createElement("object")
        annotation.appendChild(object_new)
 
        name = doc.createElement('name')
        object_new.appendChild(name)
        name_txt = doc.createTextNode(objbud[i*5])
        name.appendChild(name_txt)
 
        pose = doc.createElement('pose')
        object_new.appendChild(pose)
        pose_txt = doc.createTextNode("Unspecified")
        pose.appendChild(pose_txt)
 
        truncated = doc.createElement('truncated')
        object_new.appendChild(truncated)
        truncated_txt = doc.createTextNode("0")
        truncated.appendChild(truncated_txt)
 
        difficult = doc.createElement('difficult')
        object_new.appendChild(difficult)
        difficult_txt = doc.createTextNode("0")
        difficult.appendChild(difficult_txt)
        #threes-1#
        bndbox = doc.createElement('bndbox')
        object_new.appendChild(bndbox)
 
        xmin = doc.createElement('xmin')
        bndbox.appendChild(xmin)
        xmin_txt = doc.createTextNode(objbud[i*5+1])
        xmin.appendChild(xmin_txt)
 
        ymin = doc.createElement('ymin')
        bndbox.appendChild(ymin)
        ymin_txt = doc.createTextNode(objbud[i*5+2])
        ymin.appendChild(ymin_txt)
 
        xmax = doc.createElement('xmax')
        bndbox.appendChild(xmax)
        xmax_txt = doc.createTextNode(objbud[i*5+3])
        xmax.appendChild(xmax_txt)
 
        ymax = doc.createElement('ymax')
        bndbox.appendChild(ymax)
        ymax_txt = doc.createTextNode(objbud[i*5+4])
        ymax.appendChild(ymax_txt)
        #threee-1#
        #threee#
        
    tempfile = tmp + "test.xml"
    with open(tempfile, "w") as f:
        doc.writexml(f, addindent='  ', newl='\n', encoding='utf-8')
        # f.write(doc.toprettyxml(indent = '\t', encoding='utf-8'))
 
    rewrite = open(tempfile, "r")
    lines = rewrite.read().split('\n')
    newlines = lines[1:len(lines)-1]
    
    fw = open(wxml, "w")
    for i in range(0, len(newlines)):
        fw.write(newlines[i] + '\n')
    
    fw.close()
    rewrite.close()
    os.remove(tempfile)
    return

def parse_json(file_cur, folder):
    _, file_type = os.path.splitext(file_cur)
    if file_type == '.json':
        print('file_cur', file_cur)
        with io.open(file_cur, 'r', encoding='utf-8') as fe:#gbk
            anno = json.load(fe)
        obj_list = anno['outputs']['object']
        img_w = anno['size']['width']
        img_h = anno['size']['height']
        # img_c = anno['size']['depth']
        box_list = []
        for obj in obj_list:
            if obj['name'] != folder:
                print("obj['name'] != folder: ", obj['name'], folder)
                input()
            box_list.append(obj['name'])
            box = obj['bndbox']
            x_min = box['xmin']
            x_max = box['xmax']
            y_min = box['ymin']
            y_max = box['ymax']
            if x_min < 0:
                x_min = 0
            if y_min < 0:
                y_min = 0
            if x_max > img_w-1:
                x_max = img_w-1
            if y_max > img_h-1:
                y_max = img_h-1

            if (x_min<0 or x_min>=img_w or y_min<0 or y_min>=img_h or 
                x_max<0 or x_max>=img_w or y_max<0 or y_max>=img_h):
                print('box is error')
                input()
            box_list.append(str(x_min))
            box_list.append(str(y_min))
            box_list.append(str(x_max))
            box_list.append(str(y_max))
        return img_w, img_h, box_list
    else:
        return None, None, None


for files in os.walk(imgs_path):
    temp = "./temp"
    if not os.path.exists(temp):
        os.mkdir(temp)
    print(files)
    input()
    for file in files[2]:
        print(file + "-->start!")
        f_name, file_type = os.path.splitext(file)
        folder = files[0]
        img_name = f_name + '.jpg'
        img_path = files[0] + '/' + img_name
        json_name = files[0] + '/' + f_name + '.json'
        xml_name = files[0] + '/' + f_name + '.xml'
        print('img_path', img_name, json_name)
        im = Image.open(img_path)
        width = int(im.size[0])
        height = int(im.size[1])
        
        w, h, obj = parse_json(json_name, folder.split('/')[-1].split('-')[0])
        if w != width or h != height:
            print('w != width or h != height')
            input()
        
        writeXml(temp, img_name, width, height, obj, xml_name)
    os.rmdir(temp)