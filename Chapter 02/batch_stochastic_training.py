#----------------------------------------------------------------------------------------------------------------------------------------------
# Batch and Stochastic Training
#----------------------------------------------------------------------------------------------------------------------------------------------
#
# This python function illustrates two different training methods :  batch and stochastic training. 
# For each model, we will use  a regression model that predicts one model variable.
# We will implement a regression example in stochastic and batch training
#----------------------------------------------------------------------------------------------------------------------------------------------


# 당연한 기본 세팅 ##################################################################################
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow.python.framework import ops
ops.reset_default_graph()


# Stochastic Training #################################################################################
# Create graph  -------------------------------------------------------------------------------------------------------------------------------
sess = tf.Session()

# Create data  -------------------------------------------------------------------------------------------------------------------------------
x_vals = np.random.normal(1, 0.1, 100)
y_vals = np.repeat(10., 100)
x_data = tf.placeholder(shape=[1], dtype=tf.float32)
y_target = tf.placeholder(shape=[1], dtype=tf.float32)

# Create variable (one model parameter = A) -----------------------------------------------------------------------------------------------
A = tf.Variable(tf.random_normal(shape=[1]))

my_output = tf.mul(x_data, A)               # Add operation to graph
loss = tf.square(my_output - y_target)  # Add L2 loss operation to graph

# Initialize variables ---------------------------------------------------------------------------------------------------------------------------
init = tf.initialize_all_variables()
sess.run(init)

# Create Optimizer ---------------------------------------------------------------------------------------------------------------------------
my_opt = tf.train.GradientDescentOptimizer(0.02)
train_step = my_opt.minimize(loss)

# Sess 끝


# Run Loop ----------------------------------------------------------------------------------------------------------------------------------
# - 이 부분만 batch와 schochastic training이 다르다.
loss_stochastic = []
for i in range(100):
    rand_index = np.random.choice(100)                                                                                  # 숫자를 100중에 하나를 선택하여, 
    rand_x = [x_vals[rand_index]]                                                                                               # 이 숫자에 해당하는 x값과 y값을 뽑아낸다.
    rand_y = [y_vals[rand_index]]
    sess.run(train_step, feed_dict={x_data: rand_x, y_target: rand_y})                                          # 그리고 이 data를 feeg하여 training을 시행한다.

    if (i+1)%5==0:
        print('Step #' + str(i+1) + ' A = ' + str(sess.run(A)))
        temp_loss = sess.run(loss, feed_dict={x_data: rand_x, y_target: rand_y})
        print('Loss = ' + str(temp_loss))
        loss_stochastic.append(temp_loss)



plt.plot(range(0, 100, 5), loss_stochastic, 'b-', label='Stochastic Loss')
        


# Batch Training #################################################################################

# Re-initialize graph
ops.reset_default_graph()
sess = tf.Session()

# Declare batch size
batch_size = 20

# Create data
x_vals = np.random.normal(1, 0.1, 100)
y_vals = np.repeat(10., 100)
x_data = tf.placeholder(shape=[None, 1], dtype=tf.float32)
y_target = tf.placeholder(shape=[None, 1], dtype=tf.float32)

# Create variable (one model parameter = A)
A = tf.Variable(tf.random_normal(shape=[1,1]))

# Add operation to graph
my_output = tf.matmul(x_data, A)

# Add L2 loss operation to graph
loss = tf.reduce_mean(tf.square(my_output - y_target))

# Initialize variables
init = tf.initialize_all_variables()
sess.run(init)

# Create Optimizer
my_opt = tf.train.GradientDescentOptimizer(0.02)
train_step = my_opt.minimize(loss)

loss_batch = []
# Run Loop
for i in range(100):
    rand_index = np.random.choice(100, size=batch_size)
    rand_x = np.transpose([x_vals[rand_index]])
    rand_y = np.transpose([y_vals[rand_index]])
    sess.run(train_step, feed_dict={x_data: rand_x, y_target: rand_y})
    if (i+1)%5==0:
        print('Step #' + str(i+1) + ' A = ' + str(sess.run(A)))
        temp_loss = sess.run(loss, feed_dict={x_data: rand_x, y_target: rand_y})
        print('Loss = ' + str(temp_loss))
        loss_batch.append(temp_loss)
        
plt.plot(range(0, 100, 5), loss_stochastic, 'b-', label='Stochastic Loss')
plt.plot(range(0, 100, 5), loss_batch, 'r--', label='Batch Loss, size=20')
plt.legend(loc='upper right', prop={'size': 11})
plt.show()