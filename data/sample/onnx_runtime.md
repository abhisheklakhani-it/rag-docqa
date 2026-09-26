# ONNX Runtime

ONNX (Open Neural Network Exchange) is an open format for machine learning models. Models trained in PyTorch, TensorFlow or scikit-learn can be exported to ONNX and served with ONNX Runtime.

## Inference optimization

ONNX Runtime applies graph optimizations such as operator fusion and constant folding. Dynamic quantization converts weights to int8, which typically reduces model size by about 4x and speeds up CPU inference for transformer models.
