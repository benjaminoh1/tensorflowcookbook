# Operations on a Computational Graph

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow.python.framework import ops
ops.reset_default_graph()

# Create graph  - 항상 그렇듯이 graph를 만든다.
sess = tf.Session()

# Create tensors

# Create data to feed in
x_vals = np.array([1., 3., 5., 7., 9.])             # 일단 입력값들을 이렇게 설정해 두고
x_data = tf.placeholder(tf.float32)            # 입력값들을 넣을 placeholder들을 설정한다.
m = tf.constant(3.)

# Multiplication
prod = tf.mul(x_data, m)
for x_val in x_vals:
    print(sess.run(prod, feed_dict={x_data: x_val}))        # placeholder에 x_vals를 차례대로 feed한다.

merged = tf.merge_all_summaries()
my_writer = tf.train.SummaryWriter('C:\Users\user\Documents\GitHub\tensorflowcookbook\Chapter 02', sess.graph)