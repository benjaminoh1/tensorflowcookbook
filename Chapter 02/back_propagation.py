# Back Propagation

#############################################################################################
# This python function shows how to implement back propagation in regression and classification models.

# Regression Example:
# - We will create sample data as follows:  x-data: 100 random samples from a normal ~ N(1, 0.1)
#                                                                target: 100 values of the value 10.
# - We will fit the model:   x-data * A = target
#                                       Theoretically, A = 10.
#############################################################################################


############################### 당연한 기본 세팅 ##############################
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow.python.framework import ops
ops.reset_default_graph()


############################### Create graph ###############################
sess = tf.Session()


############################### Create data ###############################
x_vals = np.random.normal(1, 0.1, 100)              # [중앙값, 표준편차, 형태]
y_vals = np.repeat(10., 100)

x_data = tf.placeholder(shape=[1], dtype=tf.float32)
y_target = tf.placeholder(shape=[1], dtype=tf.float32)

############################### Create variable (one model parameter = A) ###############################
A = tf.Variable(tf.random_normal(shape=[1]))

############################### Add operation to graph ###############################
my_output = tf.mul(x_data, A)
loss = tf.square(my_output - y_target)   # Add L2 loss operation to graph

################################ Initialize variables ###############################
init = tf.initialize_all_variables()
sess.run(init)

############################### Create Optimizer ###############################
# - 이제 graph의 변수를 최적화하기 위해 선언을 해야 한다.
my_opt = tf.train.GradientDescentOptimizer(0.02)   # 학습률이 들어간다.
train_step = my_opt.minimize(loss)

############################### Run Loop ###############################
# - 마지막으로 우리의 training algorithm을 loop를 돌리며 여러 번 training을 거치도록 한다.
# - 여기에선 101번 loop를 돌리며, 25번째마다 결과를 표시하도록 했다.
for i in range(1000):
    rand_index = np.random.choice(100)
    rand_x = [x_vals[rand_index]]
    rand_y = [y_vals[rand_index]]
    sess.run(train_step, feed_dict={x_data: rand_x, y_target: rand_y})
    if (i+1)%25==0:
        print('Step #' + str(i+1) + ' A = ' + str(sess.run(A)))
        print('Loss = ' + str(sess.run(loss, feed_dict={x_data: rand_x, y_target: rand_y})))

# Classification Example
# We will create sample data as follows:
# x-data: sample 50 random values from a normal = N(-1, 1)
#         + sample 50 random values from a normal = N(1, 1)
# target: 50 values of 0 + 50 values of 1.
#         These are essentially 100 values of the corresponding output index
# We will fit the binary classification model:
# If sigmoid(x+A) < 0.5 -> 0 else 1
# Theoretically, A should be -(mean1 + mean2)/2

ops.reset_default_graph()

# Create graph
sess = tf.Session()

# Create data
x_vals = np.concatenate((np.random.normal(-1, 1, 50), np.random.normal(3, 1, 50)))
y_vals = np.concatenate((np.repeat(0., 50), np.repeat(1., 50)))
x_data = tf.placeholder(shape=[1], dtype=tf.float32)
y_target = tf.placeholder(shape=[1], dtype=tf.float32)

# Create variable (one model parameter = A)
A = tf.Variable(tf.random_normal(mean=10, shape=[1]))

# Add operation to graph
# Want to create the operstion sigmoid(x + A)
# Note, the sigmoid() part is in the loss function
my_output = tf.add(x_data, A)

# Now we have to add another dimension to each (batch size of 1)
my_output_expanded = tf.expand_dims(my_output, 0)
y_target_expanded = tf.expand_dims(y_target, 0)

# Initialize variables
init = tf.initialize_all_variables()
sess.run(init)

# Add classification loss (cross entropy)
xentropy = tf.nn.sigmoid_cross_entropy_with_logits(my_output_expanded, y_target_expanded)

# Create Optimizer
my_opt = tf.train.GradientDescentOptimizer(0.05)
train_step = my_opt.minimize(xentropy)

# Run loop
for i in range(100):
    rand_index = np.random.choice(100)
    rand_x = [x_vals[rand_index]]
    rand_y = [y_vals[rand_index]]
    
    sess.run(train_step, feed_dict={x_data: rand_x, y_target: rand_y})
    if (i+1)%200==0:
        print('Step #' + str(i+1) + ' A = ' + str(sess.run(A)))
        print('Loss = ' + str(sess.run(xentropy, feed_dict={x_data: rand_x, y_target: rand_y})))
        
# Evaluate Predictions
# predictions = []
# for i in range(len(x_vals)):
   # x_val = [x_vals[i]]
  # prediction = sess.run(tf.round(tf.sigmoid(my_output)), feed_dict={x_data: x_val})
 #  predictions.append(prediction[0]) 
# accuracy = sum(x==y for x,y in zip(predictions, y_vals))/100.
# print('Ending Accuracy = ' + str(np.round(accuracy, 2)))


# Graph를 눈으로 봐 보자.
file_writer = tf.summary.FileWriter('C:/Users/user/Documents/GitHub/tensorflowcookbook', sess.graph)                # sess.graph contains the graph definition; that enables the Graph Visualizer.
tensorboard --logdir=training:C:/Users/user/Documents/GitHub/tensorflowcookbook                                           # 이 부분만 cmd에서 따로 시행하여야만이 된다. Tensorflow는 :와 --를 구분 못하기 때문 
start chrome http://192.168.0.2:6006