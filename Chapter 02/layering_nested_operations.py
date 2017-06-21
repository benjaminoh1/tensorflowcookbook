# Layering Nested Operations
# - 데이터가 형태가 바뀌는 것도 중요하다. 예를 들어 3x5행렬에 5x1행렬을 곱한다음, 다시 1x1행렬을 곱한다면 최종적으로는 3x1행렬이 된다.

# 기본 세팅
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

# Computational Graph 생성
sess = tf.Session()

#Feed 할 데이터의 생성
my_array = np.array([[1., 3., 5., 7., 9.],
                                 [-2., 0., 2., 4., 6.],
                                 [-6., -3., 0., 3., 6.]])

x_vals = np.array([my_array, my_array + 1])
x_data = tf.placeholder(tf.float32, shape=(3, 5))       #혹은 필요하다면, 데이터 크기를 모를 경우 tf.placeholder(tf.float32, shape=(3,none)) 으로 해도 될 것이다.

m1 = tf.constant([[1.],[0.],[-1.],[2.],[4.]])
m2 = tf.constant([[2.]])
a1 = tf.constant([[10.]])

# 이제 operation을 해 보자 
prod1 = tf.matmul(x_data, m1)           # 1st Operation Layer = Multiplication
prod2 = tf.matmul(prod1, m2)           # 2nd Operation Layer = Multiplication
add1 = tf.add(prod2, a1)                    # 3rd Operation Layer = Addition

#마지막으로 데이터를 Feed하면 된다. 
for x_val in x_vals:
    print(sess.run(add1, feed_dict={x_data: x_val}))

merged = tf.summary.merge_all()
my_writer = tf.train.SummaryWriter('C:\Users\user\Documents\GitHub\tensorflowcookbook\Chapter 02', sess.graph)