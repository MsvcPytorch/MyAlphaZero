import tensorflow as tf
import tf2onnx
import onnx

# 1. 加载并训练您的 Keras 模型
# 这里假设您已经训练好并保存为 'red_blue_chess_evaluation_model_resnet.h5'

# 2. 加载 Keras 模型
model = tf.keras.models.load_model('model.h5')

# 3. 保存为 SavedModel 格式
model.save('saved_model')

# 4. 定义输入签名
# spec = (tf.TensorSpec((None, 6, 6, 5), tf.float32, name="board_input"),
        #tf.TensorSpec((None, 3), tf.float32, name="extra_input"))
spec = (tf.TensorSpec((None, 8, 8, 4), tf.float32, name="board_input"),)

# 5. 转换为 ONNX
output_path = "model.onnx"
model_proto, _ = tf2onnx.convert.from_keras(model, input_signature=spec, opset=13, output_path=output_path)

print(f"模型已成功转换并保存为 {output_path}")

# 6. 验证 ONNX 模型
onnx_model = onnx.load(output_path)
onnx.checker.check_model(onnx_model)
print("ONNX 模型验证成功！")
