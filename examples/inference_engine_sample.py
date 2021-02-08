from openvino.inference_engine import IENetwork, IEPlugin
import os
import numpy as np


plugin = IEPlugin(dev = args.device, plugin_dirs = args.plugin_dirs)

model_xml = args.model 
model_bin = os.path.splitext(model_xml)[0]+'.bin'

net = IENetwork(model=model_xml, weights=model_bin)

Exec_net = plugin.load(network=net)

# Read and preprocess the images
n, c, h, w = net.inputs[input_blob].shape
images = np.ndarray(shape=(n,c,h,w))
for i in range(n):
	image = cv2.imread(args.input[i])
	if image.shape[:-1] != (h,w):
		image = cv2.resize(image,(w,h))
	image = image.transpose((2,0,1))
	images[i] = image


res = Exec_net.infer(inputs={input_blob:images})

res = res[out_blob]
for i, probs in enumerate(res):
	probs = np.squeeze(probs)
	top_ind = np.argsort(probs)[-args.number_top:][::-1]
	print(f'Input {args.input[i]}\n')
	for id in top_ind:
		det_label=labels_map[id] if labels_map else "#{}".format(id)
		print("{:.7f} label {}".format(probs[id],det_label))
	print('\n')
