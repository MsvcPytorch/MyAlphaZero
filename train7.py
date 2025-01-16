import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import load_model
from tensorflow.keras import regularizers
import tensorflow as tf

# 定义棋盘尺寸和棋子类型
BOARD_SIZE = 8
# 数据加载与预处理
def load_and_preprocess_data(filename):
    # 读取CSV文件
    data = pd.read_csv(filename)
    
    # 提取棋盘状态特征
    board_columns = [col for col in data.columns if col.startswith('board_')]
    X_board = data[board_columns].values
    # 将一热编码的棋盘状态重塑为 (样本数, 8,8,4)
    X_board = X_board.reshape(-1, 8, 8, 4)
    
    # 提取标签
    y_value = data['label'].values
    policy_columns = [col for col in data.columns if col.startswith('policy_')]
    y_policy = data[policy_columns].values
    #y_policy = y_policy.reshape(-1, 7, 7, 4)  # 假设CSV中有对应的policy标签
    
    return X_board, y_value, y_policy

# 训练与评估模型
def train_and_evaluate(existing_model_path, X_board, y_value, y_policy):
    # 加载现有模型
    model = load_model(existing_model_path)
    print(f"已加载现有模型: {existing_model_path}")
    
    # 划分训练集和验证集
    Xb_train, Xb_val, yv_train, yv_val, yp_train, yp_val = train_test_split(
        X_board, y_value, y_policy, test_size=0.1, random_state=42)
    
    # 定义回调函数
    early_stop = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=1, restore_best_weights=True, verbose=1)
    reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-6, verbose=1)
    
    # 编译模型（确保与之前相同的编译配置）
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.002),
                  loss={'value_output': 'mean_squared_error',
                        'policy_output': 'categorical_crossentropy'})
    
    # 训练模型
    history = model.fit(
        {'board_input': Xb_train},
        {'value_output': yv_train, 'policy_output': yp_train},
        epochs=5,
        batch_size=512,
        validation_data=({'board_input': Xb_val},
                         {'value_output': yv_val, 'policy_output': yp_val}),
        callbacks=[early_stop, reduce_lr],
        verbose=1,
        shuffle=True
    )
    
    # 评估模型
    loss, value_loss, policy_loss = model.evaluate(
        {'board_input': Xb_val},
        {'value_output': yv_val, 'policy_output': yp_val},
        verbose=0)
    print(f'验证集总损失: {loss:.4f}')
    print(f'验证集 Value 损失: {value_loss:.4f}')
    print(f'验证集 Policy 损失: {policy_loss:.4f}')
    
    return model, history

# 主函数
def main():
    import tensorflow as tf
    
    # 设置随机种子以确保结果可重复
    np.random.seed(42)
    tf.random.set_seed(42)
    
    # 加载数据
    filename = 'data.csv'  # 请替换为您的文件名
    X_board, y_value, y_policy = load_and_preprocess_data(filename)
    
    print(f'数据集大小: {X_board.shape[0]} 样本, 8x8x4 特征')
    print(f'Value 标签大小: {y_value.shape}')
    print(f'Policy 标签大小: {y_policy.shape}')
    
    # 加载并训练模型
    existing_model_path = 'model.h5'  # 替换为您的现有模型路径
    model, history = train_and_evaluate(existing_model_path, X_board, y_value, y_policy)
    
    # 保存更新后的模型
    updated_model_path = 'nwmodel.h5'
    model.save(updated_model_path)
    print(f"更新后的模型已保存为 '{updated_model_path}'")

if __name__ == "__main__":
    main()
