import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
"""
0 = all messages are logged (default behavior)
1 = INFO messages are not printed
2 = INFO and WARNING messages are not printed
3 = INFO, WARNING, and ERROR messages are not printed
"""
import tensorflow as tf

if __name__ == '__main__':

    # To find out which devices your operations and tensors are assigned to,
    # put tf.debugging.set_log_device_placement(True) as the first statement of your program. Enabling device
    # placement logging causes any Tensor allocations or operations to be printed.

    tf.debugging.set_log_device_placement(True)
    device = tf.config.list_logical_devices()  # List Physical Devices
    print(device)

    """
    If you would like a particular operation to run on a device of your choice instead of what's automatically 
    selected for you, you can use with tf.device to create a device context, and all the operations within that 
    context will run on the same designated device.
    """

    # Place tensors on the CPU
    with tf.device('CPU'):
        a = tf.constant([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        b = tf.constant([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])

    # Run on the GPU
    c = tf.matmul(a, b)
    print(c)

    # GPU Important Points:

    # Difference between logical device and physical device
    """
    In TensorFlow, logical devices and physical devices refer to different aspects of managing computational resources,
     such as CPUs and GPUs, during the execution of a machine learning model.

Logical Devices:
    Definition: Logical devices are an abstraction provided by TensorFlow to represent a computational unit.
    Purpose: They are used to organize and manage operations in your TensorFlow computation graph.
    Example: In a model, you might specify that certain operations should run on a GPU (logical device) using tf.device('/device:GPU:0').
    
    Note: Logical devices help you express where you want operations to be executed in a high-level
    way without specifying the actual physical hardware.

Physical Devices:

    Definition: Physical devices, on the other hand, are the actual hardware components such as CPUs or GPUs.
    Purpose: They represent the real computational resources available on your machine.
    Example: If you have a machine with both a CPU and a GPU, each of them is a physical device.
    Note: TensorFlow automatically maps logical devices to available physical devices during execution 
    based on the specified device placement and the available hardware.

In summary, logical devices are a high-level abstraction that you use to express 
where operations should run in your TensorFlow graph, while physical devices are the
real hardware components that execute these operations. TensorFlow handles the mapping 
between logical and physical devices during runtime, making it easier for developers to
express their computational requirements without needing to worry about the underlying
hardware details.

    """

    # Limiting GPU memory growth
    """By default, TensorFlow maps nearly all of the GPU memory of all GPUs (subject to CUDA_VISIBLE_DEVICES) visible 
    to the process. This is done to more efficiently use the relatively precious GPU memory resources on the devices 
    by reducing memory fragmentation. To limit TensorFlow to a specific set of GPUs, 
    use the tf.config.set_visible_devices method"""
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        # Restrict TensorFlow to only use the first GPU
        try:
            tf.config.set_visible_devices(gpus[0], 'GPU')
            # Now, it will list only one GPU i.e., the first one.
            logical_gpus = tf.config.list_logical_devices('GPU')
            print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPU")
        except RuntimeError as e:
            # Visible devices must be set before GPUs have been initialized
            print(e)

    """
    In some cases it is desirable for the process to only allocate a subset of the available memory, 
    or to only grow the memory usage as is needed by the process. TensorFlow provides two methods to control this.

    The first option is to turn on memory growth by calling tf.config.experimental.set_memory_growth, which attempts to 
    allocate only as much GPU memory as needed for the runtime allocations: it starts out allocating very little memory, 
    and as the program gets run and more GPU memory is needed, the GPU memory region is extended for the TensorFlow 
    process. Memory is not released since it can lead to memory fragmentation. To turn on memory growth for a specific 
    GPU, use the following code prior to allocating any tensors or executing any ops.
    """
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        try:
            # Currently, memory growth needs to be the same across GPUs
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            logical_gpus = tf.config.list_logical_devices('GPU')
            print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
        except RuntimeError as e:
            # Memory growth must be set before GPUs have been initialized
            print(e)
