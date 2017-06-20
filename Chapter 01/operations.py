# Operations
# This function introduces various operations in Tensorflow
# Tensorflow graph에 사용할 수 있는 iperation에 대해 알아보자.

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow.python.framework import ops
ops.reset_default_graph()


sess = tf.Session()         # Open graph session

# div()
# - tensorflow에서는 div()와 그 변형함수들을 제공한다.
print(sess.run( tf.div(3,4)))                # - div()는 입력과 동일한 type을 출력한다. 예를들어 입력이 int였다면 출력도 int이다. 따라서 이 계산의 출력은 0이다.
print(sess.run( tf.truediv(3,4)))           # - truediv()는 입력과 상관없이 항상 floats를 출력한다. 따라서 출려은 0.75이다.
print(sess.run( tf.floordiv(3.0,4.0)))       # - floordiv()는 float입력에서 int출력(정확히는 계산값보다 작으면서 가장 가까운 int를 float으로)을 내놓는다. 출력은 0.0이다.

# Mod function
print(sess.run( tf.mod(22.0,5.0)))               # - mod()

# 기타 연산/함수들
print(sess.run( tf.cross([1.,0.,0.],[0.,1.,0.])))                       #  Cross Product. - cross()는 2개의 3차원 벡터에서만 정의되므로, 이 때만 사용 가능하다. 출력값은 [0 0 1]

print(sess.run( tf.sin(3.1416)))                                     # - 입력 tensor에 대한 삼각함수.
print(sess.run( tf.cos(3.1416)))                                     # - 입력 tensor에 대한 삼각함수.
print(sess.run( tf.div(tf.sin(3.1416/4.), tf.cos(3.1416/4.))))    # - 입력 tensor에 대한 삼각함수.

 # print(sess.run( tf.abs()                                                # - 절대값 
 # print(sess.run( tf.celi                                                  # - celiing function(바닥함수)
 # print(sess.run( tf.exp
 # print(sess.run( tf.floor
 # print(sess.run( tf.int
 # print(sess.run( tf.log
 # print(sess.run( tf.maximum                                           # - 원소별로 비교하여, x와 y중 가장 큰 함수를 출력으로 내보냄 
 # print(sess.run( tf.minimum
 # print(sess.run( tf.neg
 # print(sess.run( tf.pow
 # print(sess.run( tf.round
 # print(sess.run( tf.rsqrt
 # print(sess.run( tf.sign
 # print(sess.run( tf.sqrt
 # print(sess.run( tf.square


# Custom operation
test_nums = range(15)
# from tensorflow.python.ops import math_ops
# print(sess.run(tf.equal(test_num, 3)))
def custom_polynomial(x_val):
    # Return 3x^2 - x + 10
    return(tf.sub(3 * tf.square(x_val), x_val) + 10)

print(sess.run(custom_polynomial(11)))
# What should we get with list comprehension
expected_output = [3*x*x-x+10 for x in test_nums]
print(expected_output)

# Tensorflow custom function output
for num in test_nums:
    print(sess.run(custom_polynomial(num)))