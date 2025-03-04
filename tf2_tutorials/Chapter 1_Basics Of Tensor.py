import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
"""
0 = all messages are logged (default behavior)
1 = INFO messages are not printed
2 = INFO and WARNING messages are not printed
3 = INFO, WARNING, and ERROR messages are not printed
"""
import tensorflow as tf

if __name__ == '__main__':

    device = tf.config.list_logical_devices()  # List Physical Devices
    print(device)

    # Eager Constant Tensor
    tensor = tf.constant(value=[[1, 2, 3, 4], [5, 6, 7, 8]],
                         dtype=tf.float32,
                         shape=(2, 4))
    print(tensor)
    print(tensor.shape)
    print(tensor.dtype)

    # To flatten everything you can do. For specific 2d array you can also squeeze the dimension tf.squeeze(tensor)
    tensor = tf.random.normal(shape=(5, 5, 5))
    print(tf.reshape(tensor, [-1]))

    # Eager tensor cannot be modified or assigned
    # tensor[0] = [2., 3., 4., 5.]

    # Variable tensor can be modified and used when training, as they can be modified.

    variable_tensor = tf.Variable(initial_value=[[1, 2, 3], [4, 5, 6]],
                                  dtype=tf.float32,
                                  trainable=True)
    print(variable_tensor)
    print(variable_tensor.shape)
    print(variable_tensor.dtype)

    # You can modify the variable tensor like this, Note: Shape must match
    variable_tensor[0].assign([2., 3., 4.])
    variable_tensor.assign([[6, 7, 3], [4, 9, 6]])
    # Will add to the whole
    variable_tensor.assign_add([[6, 7, 3], [4, 9, 6]])
    print(variable_tensor)

    # The variable doesn't have to be a vector, it can also be a scaler.
    new_variable = tf.Variable(1.)
    new_variable.assign(value=5.)
    new_variable.assign_add(7)
    new_variable.assign_sub(2.)
    print(new_variable)
    print(new_variable.shape)

    # Some initialization techniques
    X = tf.random.uniform(shape=(2, 2), minval=0, maxval=2, dtype=tf.int32)
    Y = tf.random.normal(shape=(2, 2), mean=0, stddev=1, dtype=tf.float32)
    zeros = tf.zeros(shape=(2, 2))
    ones = tf.ones(shape=(2, 2))
    one_like = tf.ones_like(X)
    zero_like = tf.zeros_like(X)

    # Generate identity matrix, if num_rows != num_columns. Then it will put 1 in diagnol and rest 0
    identity = tf.eye(num_rows=3, num_columns=4)
    range = tf.range(start=10, limit=110, delta=10, dtype=tf.float32)
    range_shape = tf.reshape(range, shape=(1, 5, 2))

    # Reducing and Expanding dimension
    squeeze = tf.squeeze(range_shape, axis=0)
    expand = tf.expand_dims(range_shape, axis=-1)

    print(expand.shape)

    # Typecasting, must be done manually
    tensor = tf.constant([1, 2, 3], dtype=tf.int32)
    casted = tf.cast(tensor, dtype=tf.float32)
    print(tensor.shape)

    # Operations on Tensors
    tensor1 = tf.constant([[1, 2, 3], [4, 5, 6]], dtype=tf.float32)
    tensor2 = tf.constant([[0, 1, 2], [3, 4, 5]], dtype=tf.float32)

    # Element Wise
    print(tensor1 + tensor2)
    print(tensor1 - tensor2)
    print(tensor1 * tensor2)
    print(tensor1 / tensor2)

    # Or
    print(tf.add(tensor1, tensor2))
    print(tf.subtract(tensor1, tensor2))
    print(tf.multiply(tensor1, tensor2))
    print(tf.divide(tensor1, tensor2))

    # Dot Product
    # Operations on Tensors
    tensor1 = tf.constant([[1, 2, 3], [4, 5, 6]], dtype=tf.float32)
    tensor2 = tf.constant([[0, 1, 2], [3, 4, 5]], dtype=tf.float32)
    dot_product = tf.matmul(tensor1, tf.transpose(tensor2))  # Shape must match

    # Axes wise operation, reduce_sum/mean/min/max..
    row_axes = tf.reduce_sum(tensor1, axis=1)
    row_mean = tf.reduce_mean(tensor1, axis=1)
    print(row_mean)

    # More advanced product, use tf.einsum

    # Batch multiplication
    m0 = tf.random.normal(shape=[2, 2, 3])
    m1 = tf.random.normal(shape=[2, 3, 5])
    result = tf.einsum("ijk, ikl->ijl", m0, m1)

    m = tf.reshape(tf.range(9), [3, 3])

    # Get only the diagnol elements.
    # Here 'ii' represents the index where both are same.
    # Therefore, ii->i, means just return the element where both the indices are same.
    diag = tf.einsum('ii->i', m)
    print(diag)

    # Repeated indices will give you the trace of matrix
    trace = tf.einsum('ii', m)
    assert trace == sum(diag)
    print(trace.shape)

    # Transpose
    transpose = tf.einsum('ij->ji', m)
    print(transpose.shape)

    # Dot Product

    u = tf.random.normal(shape=[5])
    v = tf.random.normal(shape=[5])
    e = tf.einsum('i,i->', u, v)  # output = sum_i u[i]*v[i]
    print(e.shape)

    # Outer product
    u = tf.random.normal(shape=[3])
    v = tf.random.normal(shape=[5])
    e = tf.einsum('i,j->ij', u, v)  # output[i,j] = u[i]*v[j]
    print(e.shape)

    # Concatenation
    tensor1 = tf.constant([[1, 2, 3], [4, 5, 6]], dtype=tf.float32)
    tensor2 = tf.constant([[0, 1, 2], [3, 4, 5]], dtype=tf.float32)
    column_concatenation = tf.concat([tensor1, tensor2], axis=1)
    print(column_concatenation.shape)

    # Element access
    tensor = tf.constant([1, 2, 3, 4, 5, 6, 7, 8], dtype=tf.float32)
    print(tensor[tensor % 2 == 0])

    # Gather the indices where the condition follows
    indices = tf.squeeze(tf.where(tensor % 2 == 0))
    print(tf.gather(tensor, indices))
