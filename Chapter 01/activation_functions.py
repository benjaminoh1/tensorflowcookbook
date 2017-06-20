# Activation Functions(활성함수)
# This function introduces activation functions in Tensorflow

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import tensorflow.nn as nn
from tensorflow.python.framework import ops
ops.reset_default_graph()

# Open graph session
sess = tf.Session()                                               # 항상 그렇듯이 이렇게 tensorflow graph를 실행해야 한다.

x_vals = np.linspace(start=-10., stop=10., num=100)

# ReLU activation       - 가장 일반적이고 널리 쓰임, max(0,x)와 같음
print(sess.run(tf.nn.relu([-3., 3., 10.])))                         
y_relu = sess.run(tf.nn.relu(x_vals))

# ReLU-6 activation         - min(max(0,x), 6)와 같음. CNN과 RNN에서 자주 쓰이는 함수 중 하나. 출력이 [0 3 6]이 된다.
print(sess.run(tf.nn.relu6([-3., 3., 10.])))
y_relu6 = sess.run(tf.nn.relu6(x_vals))

# Sigmoid activation          - smooth activation 중에서 가장 많이 사용되며 logistic function이라고도 불린다. 하지만 경사소실이 잦다. 0~1의 치역
print(sess.run(tf.nn.sigmoid([-1., 0., 1.])))
y_sigmoid = sess.run(tf.nn.sigmoid(x_vals))

# Hyper Tangent activation  - sigmoid와 같은 성ㅅ질을 가지지만, -1~1의 치역
print(sess.run(tf.nn.tanh([-1., 0., 1.])))
y_tanh = sess.run(tf.nn.tanh(x_vals))

# Softsign activation           - sign 함수의 부드러운 버젼. x / (abs(x)+1), -1~1의 치역
print(sess.run(tf.nn.softsign([-1., 0., 1.])))
y_softsign = sess.run(tf.nn.softsign(x_vals))

# Softplus activation           - ReLU 함수의 부드러운 버젼. log( exp(x)+1 ). 0~1의 치역
print(sess.run(tf.nn.softplus([-1., 0., 1.])))
y_softplus = sess.run(tf.nn.softplus(x_vals))

# Exponential linear activation(ELU)     - softplus와 비슷하지만, -1~1의 치역을 가진다. x <0 일 때 exp(x)+1, x >0 일 때 x
print(sess.run(tf.nn.elu([-1., 0., 1.])))
y_elu = sess.run(tf.nn.elu(x_vals))

# Plot the different functions
plt.plot(x_vals, y_softplus, 'r--', label='Softplus', linewidth=2)
plt.plot(x_vals, y_relu, 'b:', label='ReLU', linewidth=2)
plt.plot(x_vals, y_relu6, 'g-.', label='ReLU6', linewidth=2)
plt.plot(x_vals, y_elu, 'k-', label='ExpLU', linewidth=0.5)
plt.ylim([-1.5,7])
plt.legend(loc='top left')
plt.show()

plt.plot(x_vals, y_sigmoid, 'r--', label='Sigmoid', linewidth=2)
plt.plot(x_vals, y_tanh, 'b:', label='Tanh', linewidth=2)
plt.plot(x_vals, y_softsign, 'g-.', label='Softsign', linewidth=2)
plt.ylim([-2,2])
plt.legend(loc='top left')
plt.show()
