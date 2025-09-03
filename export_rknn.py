import torch
import PIL
import PIL.Image
import numpy as np
from rknn.api import RKNN
import argparse

def get_args():
    p = argparse.ArgumentParser()
    p.add_argument("--onnx_path", type=str, default="/home/saul/code/rknn_python/model/pfld.onnx")
    p.add_argument("--rknn_path", type=str, default="/home/saul/code/rknn_python/model/pfld.rknn")
    # these settings are dependent with model, should align with the model in traning stage.
    p.add_argument("--mean", type=list, default=[0., 0., 0.])
    p.add_argument("--std", type=list, default=[255.0, 255.0, 255.0])
    p.add_argument("--rgb2bgr", type=bool, default=True)
    p.add_argument("--chip", type=str, default="RV1106")
    p.add_argument("--alg", type=str, default="normal", choices=["normal", "mmse"])
    return p

parser = get_args()
args = parser.parse_args()
# config, load, build and export RKNN
rknn_runtime = RKNN(verbose=True)
mean=args.mean
std=args.std
rknn_runtime.config(
    mean_values=[[v for v in mean]],
    std_values=[[v for v in std]],
    quant_img_RGB2BGR=args.rgb2bgr,
    target_platform=args.chip,
    # MMSE is better than normal, but need more time and memory.
    # It set MMSE, the length of dataset should not exceed 100! If not, there will be OOM.
    quantized_algorithm=args.alg,
    quantized_method="channel",
)

ret = rknn_runtime.load_onnx(model=args.onnx_path)
if ret != 0:
    raise TypeError(f"model load error, {ret}")

ret = rknn_runtime.build(do_quantization=True, dataset="./dataset.txt")
if ret != 0:
    raise TypeError(f"rknn build error, ret: {ret}")

ret = rknn_runtime.export_rknn(args.rknn_path)
if ret != 0 :
    raise TypeError(f"export RKNN model error, ret: {ret}")

# init rknn runtime
ret = rknn_runtime.init_runtime(target=None)
if ret != 0:
    raise Exception(f"rknn init fail: {ret}")

rknn_runtime.release()