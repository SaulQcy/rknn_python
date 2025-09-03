import torch
import PIL
import PIL.Image
import numpy as np
from rknn.api import RKNN
import argparse
import cv2

def get_args():
    p = argparse.ArgumentParser()
    p.add_argument("--onnx", type=str, default="model/mobilenetv2-12.onnx")
    p.add_argument("--rknn", type=str, default="model/mobilenetv2-12.rknn")
    p.add_argument("--image", type=str, default="cat_224x224.jpg")
    # these settings are dependent with model, should align with the model in traning stage.
    p.add_argument("--mean", type=list, default=[0., 0., 0.])
    p.add_argument("--std", type=list, default=[255.0, 255.0, 255.0])
    p.add_argument("--rgb2bgr", type=bool, default=False)
    p.add_argument("--chip", type=str, default="RV1106")
    return p

parser = get_args()
args = parser.parse_args()
# config, load, build and export RKNN
rknn = RKNN(verbose=True)
mean=args.mean
std=args.std
rknn.config(
    mean_values=[[v for v in mean]],
    std_values=[[v for v in std]],
    quant_img_RGB2BGR=args.rgb2bgr,
    target_platform=args.chip,
    # MMSE is better than normal, but need more time and memory.
    # It set MMSE, the length of dataset should not exceed 100! If not, there will be OOM.
    # quantized_algorithm="mmse",
    quantized_algorithm="normal",
    quantized_method="channel",
    optimization_level=3,
    quantized_dtype="asymmetric_quantized-8",
)

ret = rknn.load_onnx(model=args.onnx)
if ret != 0:
    raise TypeError(f"model load error, {ret}")

ret = rknn.build(do_quantization=True, dataset="./dataset.txt")
if ret != 0:
    raise TypeError(f"rknn build error, ret: {ret}")

ret = rknn.export_rknn(args.rknn)
if ret != 0 :
    raise TypeError(f"export RKNN model error, ret: {ret}")

# init rknn runtime
ret = rknn.init_runtime(target=None)
if ret != 0:
    raise Exception(f"rknn init fail: {ret}")

IMAGE_PATH = args.image
img_ori = cv2.imread(IMAGE_PATH)
img_ori = cv2.cvtColor(img_ori, cv2.COLOR_BGR2RGB)
img_ori = cv2.resize(img_ori, (224, 224), interpolation=cv2.INTER_LINEAR)
img = np.expand_dims(img_ori, 0)

outputs = rknn.inference(inputs=[img])              # 一般返回 float32（已反量化）
logits = outputs[0].reshape(-1)                     # (1000,)

# softmax（数值稳定版）
logits = logits - logits.max()
expv = np.exp(logits)
probs = expv / expv.sum()

topk = probs.argsort()[-10:][::-1]
for i, idx in enumerate(topk, 1):
    print(f'{i:2d}. idx={idx:4d} prob={probs[idx]:.4f}  logit={logits[idx]+logits.max():.4f}')
rknn.release()