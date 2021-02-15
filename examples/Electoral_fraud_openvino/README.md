## Electoral_Fraud_openvino

## Steps to convert model: 

- `git clone git@github.com:kairavkkp/Electoral-Fraud-Detection.git`
- `git clone git@github.com:mystic123/tensorflow-yolo-v3.git`
- `cd tensorflow-yolo-v3`
- `git checkout ed60b90`
- `python3 convert_weights_pb.py --class_names person-election-obj.names --data_format NHWC --weights_file person-election-yolov3_2000.weights`
- `python /opt/intel/openvino_2020.3.341/deployment_tools/model_optimizer/mo_tf.py --input_model frozen_darknet_yolov3_model.pb  --tensorflow_use_custom_operations_config /opt/intel/openvino_2020.3.341/deployment_tools/model_optimizer/extensions/front/tf/yolo_v3.json  --batch 1`

XML and BIN Files generated.

## Steps to inference: 
- `python object_detection_sample.py -i test.png -m frozen_darknet_yolov3_model.xml`
- Inference Output present in `inference.log`