# PipeMFL-240K: A Large-scale Dataset and Benchmark for Object Detection in Pipeline Magnetic Flux Leakage Imaging #
https://arxiv.org/abs/2602.07044
Suggested Citation: T. Qu, S. Yang, H. Wang et al., “PipeMFL-240K: A Large-scale Dataset and Benchmark for Object Detection in Pipeline Magnetic Flux Leakage Imaging,” arXiv.org, 2026. https://arxiv.org/abs/2602.07044.
# 1. Dataset
https://doi.org/10.57967/hf/7651
# 2. Train #
## retina ##
python retina_train.py --train-root /root//train  --val-root /root/val 
## fasterrcnn ##
python fasterrcnn_train.py --train-root /root/a/train --val-root /root/val 
## yolov5, yolov8, yolo11 ##  
python yolov8_train.py --data /root/data.yaml --workers 16 --batch 256
## yoloworld ## 
python yoloworld_train --data /root/data.yaml --workers 16 --batch 256
## yolo26 ## 
python yolo26_train.py --data /root/data.yaml --workers 16 --batch 256
## rtdetr ## 
python rtdetr_train.py --data /root/data.yaml --workers 16 --batch 256
## rfdetr ##
python rfdetr_train.py
# 3. Metrics, Infer #
## retina ##
python metrics_retina.py
## fasterrcnn ##
python metrics_fasterrcnn.py
## yolov5, yolov8, yolo11, yoloworld ##
python metrics_yolo.py
## yolo26 ##
python metrics_yolo26.py
## rtdetr ##
python metrics_rtdetr.py
## rfdetr ##
python metrics_rfdetr.py
# 4. AP50 AP50:95 P R F1 summary per class #
metrics_cal.ipynb
# 5. Test image #
python image_test.py --model /root/xxx.pt  --image /root/testimage/test3.png  --output /root/output
# 6. Subset generating #
yolo
python subset_generate_yolo.py --data data.yaml --ratio 8
coco
python subset_generate_rfdetr.py --data_root /root/train  --ratio 8 --outdir /root/output
