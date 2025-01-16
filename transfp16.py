import onnx
from onnxconverter_common import float16

# Load the original FP32 ONNX model
model_fp32_path = 'model.onnx'  # Replace with your model's path
model = onnx.load(model_fp32_path)

# Convert the model to FP16
# The keep_io_types parameter ensures that input and output tensor types remain FP32
model_fp16 = float16.convert_float_to_float16(model, keep_io_types=True)

# Save the FP16 model
model_fp16_path = 'model_fp16.onnx'  # Specify the output path
onnx.save(model_fp16, model_fp16_path)

print(f"Converted FP32 model '{model_fp32_path}' to FP16 model '{model_fp16_path}'")
