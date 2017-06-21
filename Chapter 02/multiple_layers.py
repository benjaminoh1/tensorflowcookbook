# Layering Nested Operations & graph visualization
# - 이제 각 layer들마다 어떻게 data를 이동시킬 지 연습해 보도록 하겠다.
# - 입력으로 작은 사이즈의 2D 이미지가 들어온다고 하고 시작해보자.

# 기본 세팅 - tf와 Numpy를 불러온다.
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow.python.framework import ops
ops.reset_default_graph()

# Computational Graph 생성
sess = tf.Session()


# 일단 입력으로 쓰일 2D 4x4이미지를 만든다. [1,4,4,1]=[이미지번호, 높이, 폭, 채널]을 이야기한다. Tensorflow에서는 이런 식으로 이미지를 표현한다.
x_shape = [1, 4, 4, 1]
x_val = np.random.uniform(size=x_shape)


#이 data를 받을 placeholder를 설정한다.
x_data = tf.placeholder(tf.float32, shape=x_shape)


# 입력 4x4 이미지를 2칸 stride로 움직이며 평균을 내는 2x2 window를 만들기 위하여, conv2d()를 사용하겠다. 
# 더불어 4개의 2x2이미지의 평균을 내기 위하여, 상수 0.25를 곱하겠다. 
my_filter = tf.constant(0.25, shape=[2, 2, 1, 1])                           # 상수를 만드는 부분. 0.25로 가득 찬, [filter_height, filter_width, in_channels, out_channels]을 만들겠다는 거다.
my_strides = [1, 2, 2, 1]                                                               # strides는 2칸 2칸씩 움직이겠다는 이야기. Strides는 [1, stride, stride, 1]로 항상 만들어야 한다.
mov_avg_layer= tf.nn.conv2d(x_data, my_filter, my_strides, padding='SAME', name='Moving_Avg_Window')


# Define a custom layer which will be sigmoid(Ax+b) where
# x is a 2x2 matrix and A and b are 2x2 matrices
def custom_layer(input_matrix):
    input_matrix_sqeezed = tf.squeeze(input_matrix)                 # shape(squeeze(t))는 size가 1인 dimension을 제거한다. 예를 들어 [1,4,4,1]->[4,4]로 만든다. 여기에선 이미지의 픽셀부분만 남기겠다는 거다.
    A = tf.constant([[1., 2.], [-1., 3.]])
    b = tf.constant(1., shape=[2, 2])
    temp1 = tf.matmul(A, input_matrix_sqeezed)
    temp = tf.add(temp1, b) # Ax + b
    return(tf.sigmoid(temp))                                                        # 마지막으로 sigmoid function을 적용하였다.


# Comp. graph에 위에 만든 layer를 넣어보자.
with tf.name_scope('Custom_Layer') as scope:
    custom_layer1 = custom_layer(mov_avg_layer)


# 마지막으로 session을 시행하면 된다.
print(sess.run(custom_layer1, feed_dict={x_data: x_val}))


# Graph를 눈으로 봐 보자.
file_writer = tf.summary.FileWriter('C:/Users/user/Documents/GitHub/tensorflowcookbook', sess.graph)                # sess.graph contains the graph definition; that enables the Graph Visualizer.
tensorboard --logdir=training:C:/Users/user/Documents/GitHub/tensorflowcookbook                                           # 이 부분만 cmd에서 따로 시행하여야만이 된다. Tensorflow는 :와 --를 구분 못하기 때문 
