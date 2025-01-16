import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Conv2D, Dropout, Flatten, concatenate, BatchNormalization, Activation, Add, GlobalAveragePooling2D, Reshape
from tensorflow.keras import regularizers

# 定义棋盘尺寸和棋子类型
BOARD_SIZE = 8

# 定义残差块
def residual_block(x, filters, kernel_size=3, stride=1, l2_reg=1e-4):
    shortcut = x
    
    # 第一层卷积
    x = Conv2D(filters, kernel_size=kernel_size, strides=stride, padding='same',
               kernel_regularizer=regularizers.l2(l2_reg))(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    
    # 第二层卷积
    x = Conv2D(filters, kernel_size=kernel_size, strides=1, padding='same',
               kernel_regularizer=regularizers.l2(l2_reg))(x)
    x = BatchNormalization()(x)
    
    # 跳跃连接调整
    if shortcut.shape[-1] != filters or stride != 1:
        shortcut = Conv2D(filters, kernel_size=1, strides=stride, padding='same',
                          kernel_regularizer=regularizers.l2(l2_reg))(shortcut)
        shortcut = BatchNormalization()(shortcut)
        
    # 添加跳跃连接并激活
    x = Add()([x, shortcut])
    x = Activation('relu')(x)
    
    return x
def value_head(input):
    conv1 = Conv2D(kernel_size=1,
                strides=1,
                filters=1,
                padding="same")(input)

    bn1 = BatchNormalization()(conv1)
    bn1_relu = Activation('relu')(bn1)
    
    flat = Flatten()(bn1_relu)

    dense1 = Dense(128)(flat)
    dn_relu = Activation('relu')(dense1)
    dn_relu = Dropout(0.3)(dn_relu)

    dense2 = Dense(128)(dn_relu)
    dense2 = Dropout(0.3)(dense2)

    return dense2

def policy_head(input):
    conv1 = Conv2D(kernel_size=2,
                strides=1,
                filters=1,
                padding="same")(input)
    bn1 = BatchNormalization()(conv1)
    bn1_relu = Activation('relu')(bn1)
    flat = Flatten()(bn1_relu)
    return flat

# 构建 Policy-Value ResNet 模型
def build_policy_value_resnet_model(input_shape_board, num_residual_blocks=6, filters=192, l2_reg=1e-4):
    # 输入层 - 棋盘状态
    input_board = Input(shape=input_shape_board, name='board_input')
    
    # 初始卷积层
    x = Conv2D(filters, kernel_size=3, strides=1, padding='same',
    kernel_regularizer=regularizers.l2(l2_reg))(input_board)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    # 添加多个残差块
    for _ in range(num_residual_blocks):
        x = residual_block(x, filters=filters, kernel_size=3, stride=1, l2_reg=l2_reg)
    
    policy_output = Dense(8*8*3, activation='softmax', name='policy_output')(policy_head(x))
    value_output = Dense(1, activation='tanh', name='value_output')(value_head(x))
    
    # 构建模型
    model = Model(inputs=[input_board], outputs=[value_output, policy_output])
    
    return model

# 构建并保存初始模型
def create_initial_model():
    model = build_policy_value_resnet_model(input_shape_board=(BOARD_SIZE, BOARD_SIZE, 4),
                                           num_residual_blocks=6, filters=96, l2_reg=1e-4)
    
    # 显示模型摘要
    model.summary()
    
    # 保存模型为H5格式
    initial_model_path = '6b96f.h5'
    model.save(initial_model_path)
    print(f"未训练的模型已保存为 '{initial_model_path}'")

if __name__ == "__main__":
    create_initial_model()
