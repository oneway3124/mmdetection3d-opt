import tensorflow as tf

# 定义输入数据和卷积核
input_data = tf.Variable(tf.random_normal([1, 28, 28, 3]))  # 输入数据形状为[batch_size, height, width, channels]
filter_data = tf.Variable(tf.random_normal([3, 3, 3, 16]))  # 卷积核形状为[filter_height, filter_width, input_channels, output_channels]

# 定义卷积层
conv_layer = tf.nn.conv2d(input_data, filter_data, strides=[1, 2, 2, 1], padding='SAME')

# 运行计算图
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    conv_result = sess.run(conv_layer)
    print(conv_result.shape)
