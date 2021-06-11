import sys
import onnx
import os
import argparse
import numpy as np
import cv2
import onnxruntime
import torch

import utils.gpu as gpu
from utils.data_augment import Resize

from model.build_model import Build_Model
from eval.evaluator import Evaluator
import config.yolov4_config as cfg

def post_processing(pred_bbox, evaluator, org_h, org_w, valid_scale=(0, np.inf)):
    bboxes = evaluator.convert_pred(
            pred_bbox, evaluator.val_shape, (org_h, org_w), valid_scale
        )
    print('sppppppp', bboxes.shape, bboxes)
    return bboxes

def plot_boxes_cv2(img_s, bboxes_prd):
    for bbox in bboxes_prd:
        coor = np.array(bbox[:4], dtype=np.int32)
        score = bbox[4]
        class_ind = int(bbox[5])
        
        classes = cfg.Customer_DATA["CLASSES"]
        class_name = classes[class_ind]
        score = "%.4f" % score
        xmin, ymin, xmax, ymax = map(str, coor)
        cv2.rectangle(img_s, (int(xmin), int(ymin)), (int(xmax), int(ymax)), (0,255,0), 1, 8, 0)
        height = int(ymax) - int(ymin)
        bottomLeftCornerOfText = (int(xmin), int(int(ymin)+0.1*height))
        fontScale = 1.0
        fontColor = (255, 200, 0)
        lineType = 2
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(img_s, str(score), bottomLeftCornerOfText, 
            font, fontScale, fontColor, lineType)
        bottomLeftCornerOfText = (int(xmin), int(ymax))
        cv2.putText(img_s, str(class_name), bottomLeftCornerOfText, 
            font, fontScale, fontColor, lineType)
    cv2.imwrite('./onnx-test.jpg', img_s)

def detect(session, image_src, test_shape):
    IN_IMAGE_H = session.get_inputs()[0].shape[2]
    IN_IMAGE_W = session.get_inputs()[0].shape[3]
    
    img_in = Resize((test_shape, test_shape), correct_box=False)(
            image_src, None
        ).transpose(2, 0, 1)
    img_in = np.expand_dims(img_in, axis=0).astype(np.float32)

    # Input
    # resized = cv2.resize(image_src, (IN_IMAGE_W, IN_IMAGE_H), interpolation=cv2.INTER_LINEAR)
    # img_in = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
    # img_in = np.transpose(img_in, (2, 0, 1)).astype(np.float32)
    # img_in = np.expand_dims(img_in, axis=0)
    # img_in /= 255.0
    print("Shape of the network input: ", img_in.shape, type(img_in), img_in.dtype)
    input()
    # Compute
    input_name = session.get_inputs()[0].name

    outputs = session.run(None, {input_name: img_in})
    print('list ', type(outputs), len(outputs))
    pred_tensor0 = outputs[0] #list member 0 ; img 0
    pred_tensor1 = outputs[1] #list member 0 ; img 0
    pred_tensor2 = outputs[2]
    pred_tensor3 = outputs[3]
    if type(pred_tensor3).__name__ != 'ndarray':
        print('conv ss')
        pred_tensor3 = pred_tensor3.cpu().detach().numpy()

    if type(pred_tensor0).__name__ != 'ndarray':
        print('conv ss')
        pred_tensor0 = pred_tensor0.cpu().detach().numpy()

    if type(pred_tensor1).__name__ != 'ndarray':
        print('conv ss')
        pred_tensor1 = pred_tensor1.cpu().detach().numpy()
    if type(pred_tensor2).__name__ != 'ndarray':
        print('conv ss')
        pred_tensor2 = pred_tensor2.cpu().detach().numpy()
    print("Shape of the network output: ", pred_tensor0.shape, pred_tensor1.shape, pred_tensor2.shape, pred_tensor3.shape)
    return pred_tensor3

def transform_to_onnx(weight_file, batch_size, n_classes, IN_IMAGE_H, IN_IMAGE_W):
    device = gpu.select_device(id=0)

    model = Build_Model().to(device)
    #model = Build_Model(weight_path=weight_file, resume=False).to(device)

    pretrained_dict = torch.load(weight_file, map_location=device) #torch.device('cuda')
    model.load_state_dict(pretrained_dict)

    evaluator = Evaluator(model, showatt=False)

    input_names = ["input"]
    output_names = ['boxes', 'confs']

    dynamic = False
    if batch_size <= 0:
        dynamic = True

    if dynamic:
        x = torch.randn((1, 3, IN_IMAGE_H, IN_IMAGE_W), requires_grad=True)
        onnx_file_name = "yolov4_-1_3_{}_{}_dynamic.onnx".format(IN_IMAGE_H, IN_IMAGE_W)
        dynamic_axes = {"input": {0: "batch_size"}, "boxes": {0: "batch_size"}, "confs": {0: "batch_size"}}
        # Export the model
        print('Export the onnx model ...')
        torch.onnx.export(model,
                          x,
                          onnx_file_name,
                          export_params=True,
                          opset_version=11,
                          do_constant_folding=True,
                          input_names=input_names, output_names=output_names,
                          dynamic_axes=dynamic_axes)

        print('Onnx model exporting done')
        return onnx_file_name
    else:
        x = torch.randn((batch_size, 3, IN_IMAGE_H, IN_IMAGE_W), requires_grad=True)
        x = x.to(device)
        onnx_file_name = "yolov4_{}_3_{}_{}_static.onnx".format(batch_size, IN_IMAGE_H, IN_IMAGE_W)
        # Export the model
        print('Export the onnx model ...')
        torch.onnx.export(model,
                          x,
                          onnx_file_name,
                          export_params=True,
                          opset_version=11,
                          do_constant_folding=True,
                          input_names=input_names, output_names=output_names,
                          dynamic_axes=None)

        print('Onnx model exporting done')
        return onnx_file_name, evaluator
    
def main(weight_file, image_path, batch_size, n_classes, IN_IMAGE_H, IN_IMAGE_W):
    if batch_size <= 0:
        onnx_path_demo = transform_to_onnx(weight_file, batch_size, n_classes, IN_IMAGE_H, IN_IMAGE_W)
    else:
        # Transform to onnx as specified batch size
        transform_to_onnx(weight_file, batch_size, n_classes, IN_IMAGE_H, IN_IMAGE_W)
        # Transform to onnx for demo
        onnx_path_demo, evaluator = transform_to_onnx(weight_file, 1, n_classes, IN_IMAGE_H, IN_IMAGE_W)

    session = onnxruntime.InferenceSession(onnx_path_demo)
    # session = onnx.load(onnx_path)
    print("The model expects input shape: ", session.get_inputs()[0].shape)

    image_src = cv2.imread(image_path)
    org_h = image_src.shape[0]
    org_w = image_src.shape[1]
    pred_tensor = detect(session, image_src, evaluator.val_shape)

    boxes = post_processing(pred_tensor, evaluator, org_h, org_w)

    plot_boxes_cv2(image_src, boxes)

if __name__ == '__main__':
    print("Converting to onnx and running demo ...")
    if len(sys.argv) == 7:
        
        weight_file = sys.argv[1]
        image_path = sys.argv[2]
        batch_size = int(sys.argv[3])
        n_classes = int(sys.argv[4])
        IN_IMAGE_H = int(sys.argv[5])
        IN_IMAGE_W = int(sys.argv[6])

        main(weight_file, image_path, batch_size, n_classes, IN_IMAGE_H, IN_IMAGE_W)
    else:
        print('Please run this way:\n')
        print('  python demo_onnx.py <weight_file> <image_path> <batch_size> <n_classes> <IN_IMAGE_H> <IN_IMAGE_W>')