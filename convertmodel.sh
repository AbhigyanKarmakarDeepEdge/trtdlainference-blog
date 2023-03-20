/usr/src/tensorrt/bin/trtexec --onnx=data/onnx/age_googlenet.onnx --fp16 --saveEngine=G1G2GPU.trt
/usr/src/tensorrt/bin/trtexec --onnx=data/onnx/G1.onnx --fp16 --saveEngine=G1DLA.trt --useDLACore=0
/usr/src/tensorrt/bin/trtexec --onnx=data/onnx/G2.onnx --fp16 --saveEngine=G2GPU.trt
mv G1G2GPU.trt G1DLA.trt G2GPU.trt data/trt/
