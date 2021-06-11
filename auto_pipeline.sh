python3 split_dataset_into_train_test.py

python3 utils/custom_data.py

python3 utils/json_2_xml.py

find /home/chenp/YOLOv4-pytorch/qixing-data/test -name "*.xml" | xargs -i cp {} /home/chenp/YOLOv4-pytorch/data/VOCtest-2007/VOCdevkit/VOC2007/Annotations

find /home/chenp/YOLOv4-pytorch/qixing-data/test -name "*.jpg" | xargs -i cp {} /home/chenp/YOLOv4-pytorch/data/VOCtest-2007/VOCdevkit/VOC2007/JPEGImages