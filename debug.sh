python -m debugpy --listen 5678 --wait-for-client ./tools/deploy.py \
/project/mmdeploy-ncnn/mmdeploy/configs/mmdet/detection/single-stage_ncnn_static-800x1344.py \
/project/mmdeploy-ncnn/mmdetection/configs/yolo/yolov3_d53_8xb8-320-273e_coco.py \
"/project/mmdeploy-ncnn/mmdeploy_checkpoints/mmdet/yolov3/yolov3_d53_320_273e_coco-421362b6.pth" \
"../mmdetection/demo/demo.jpg"  \
--work-dir "../mmdeploy_regression_working_dir/mmdet/yolov3/ncnn/static/fp32/yolov3_d53_320_273e_coco-421362b6"  \
--device cpu  \
--log-level INFO \
--test-img ./tests/data/tiger.jpeg


python -m debugpy --listen 5678 --wait-for-client tools/test.py \
/project/mmdeploy-ncnn/mmdeploy/configs/mmdet/detection/single-stage_ncnn_static-800x1344.py \
/project/mmdeploy-ncnn/mmdetection/configs/yolo/yolov3_d53_8xb8-320-273e_coco.py \
--model "../mmdeploy_regression_working_dir/mmdet/yolov3/ncnn/static/fp32/yolov3_d53_320_273e_coco-421362b6/end2end.param" "../mmdeploy_regression_working_dir/mmdet/yolov3/ncnn/static/fp32/yolov3_d53_320_273e_coco-421362b6/end2end.bin" \
--speed-test


python3 ./tools/test.py \
/project/mmdeploy-ncnn/mmdeploy/configs/mmdet/detection/single-stage_ncnn_static-800x1344.py \
/project/mmdeploy-ncnn/mmdetection/configs/retinanet/retinanet_r50_fpn_1x_coco.py \
--model "../mmdeploy_regression_working_dir/mmdet/retinanet/ncnn/static/fp32/retinanet_r50_fpn_1x_coco_20200130-c2398f9e"  \
