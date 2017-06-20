# Matrices and Matrix Operations
# This function introduces various ways to create matrices and how to use them in Tensorflow

import numpy as np
import tensorflow as tf
from tensorflow.python.framework import ops
ops.reset_default_graph()


sess = tf.Session()

# 2차원 매트릭스는 Numpy arrays를 통하여 만들 수 있다. 혹은 앞에서 말한 데로 tf.zeros(), tf.ones(), tf.truncated_normal() 등으로 만들 수 있다.
identity_matrix = tf.diag([1.0,1.0,1.0])     # 이런 식으로 대각행렬을 만들 수 있다.
A = tf.truncated_normal([2,3])              # 
B = tf.fill([2,3], 5.0)                             # 5로 가득찬 2x3 행렬을 만드는 방법.
C = tf.random_uniform([3,2])                # 0~1범위 내 랜덤한 숫자로 가득 찬 3x2행렬을 만드는 방법
D = tf.convert_to_tensor(np.array([[1., 2., 3.], [-3., -7., -1.], [0., 5., -2.]]))  # Np.array로부터 행렬을 만드는 방법.

print(sess.run(identity_matrix))
print(sess.run(A))
print(sess.run(B))
print(sess.run(C))                              #이런 경우에 sess.run(C)을 reinitialize한다면, 다른 랜덤한 숫자로 가득 찬다는 것을 기억해두자. 
print(sess.run(D))


# 행렬의 연산
print(sess.run(A+B))                                   #행렬합
print(sess.run(B-B))                                    #행렬차
print(sess.run(tf.matmul(B, identity_matrix)))    #행렬곱

#행렬함수들
print(sess.run(tf.transpose(C)))                       #  대각행렬 만들기. 앞에서 말한 데로 C를 reinitialize하였으므로, 앞과 다른 랜덤한 숫자로 가득 찬다.
print(sess.run(tf.matrix_determinant(D)))           # Determinant
print(sess.run(tf.matrix_inverse(D)))                   # Matrix Inverse
print(sess.run(tf.cholesky(identity_matrix)))           # Cholesky Decomposition
print(sess.run(tf.self_adjoint_eig(D)))                     # Eigenvalues and Eigenvectors. 연산하고 나면 1행들에는 eigenvalues가, 나머지 행들에는 해당 eigenvector가 출력된다.

