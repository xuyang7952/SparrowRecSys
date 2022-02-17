
import tensorflow as tf
#载入MINST数据集
mnist = tf.keras.datasets.mnist
#划分训练集和测试集
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0


#定义模型结构和模型参数
model = tf.keras.models.Sequential([
    #输入层28*28维矩阵,flatten,展平，三层神经网络：28*28 的输入层，128 维的隐层，由 softmax 构成的多分类输出层
    tf.keras.layers.Flatten(input_shape=(28, 28)), 
    #128维隐层，使用relu作为激活函数
    tf.keras.layers.Dense(128, activation='relu'),
    # 设置dropout=0.2，防止过拟合
    tf.keras.layers.Dropout(0.2),
    #输出层采用softmax模型，处理多分类问题
    tf.keras.layers.Dense(10, activation='softmax')
])
#定义模型的优化方法(adam)，损失函数(sparse_categorical_crossentropy)和评估指标(accuracy)
# optimizer 指的是模型训练的方法，adam常用，随机梯度下降（SGD），自动变更学习率的 AdaGrad，或者动量优化 Momentum 等
# loss 指的是损失函数，使用多分类交叉熵（Sparse Categorical Crossentropy）作为损失函数。
# epochs 指的是训练时迭代训练的次数，单次训练是模型训练过程把所有训练数据学习了一遍，epochs=5模型要反复5次学习训练数据，以便模型收敛
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])


#训练模型，进行5轮迭代更新(epochs=5）
model.fit(x_train, y_train, epochs=5)
#评估模型
model.evaluate(x_test,  y_test, verbose=2)