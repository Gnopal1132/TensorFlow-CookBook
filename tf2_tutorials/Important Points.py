import tensorflow as tf
import cv2


if __name__ == '__main__':

    # Point 1: Difference between convert_image_type, Normalization and tf.cast
    image_uint8 = tf.constant([[[255, 128, 64], [32, 0, 128]]], dtype=tf.uint8)

    """
    Note, tf.image.convert_image_type: is specifically designed for converting images between different data
    types. It ensures that the conversion respects the range and scale of image data.

    Automatic Rescaling: When converting between different types, it automatically rescales the pixel values
    appropriately. For example, if you convert from tf.uint8 (values in the range 0-255) to tf.float32, it will rescale
    the values to the range 0-1. So further normalization is not required.

    Handling Different Image Types: It handles the conversion correctly for image types like
    uint8, uint16, float16, float32, etc.
    
    tf.cast() is general approach to simply just convert the type of a number.
    """

    # Using tf.image.convert_image_type to convert to tf.float32
    converted_image = tf.image.convert_image_dtype(image_uint8, tf.float32)

    print("Normalizing Image", tf.divide(tf.cast(image_uint8, tf.float32), 255.0))

    # Print the result
    print("Converted image:", converted_image)
    print("Converted image dtype:", converted_image.dtype)
    print("Converted image value range:", tf.reduce_min(converted_image).numpy(), "-",
          tf.reduce_max(converted_image).numpy())

    # Point 2: Understanding Einstein Notation

    """
    Key Concepts to Understand:
    
    1. Indices Represent Dimensions: Each letter (index) represents a dimension of the tensor.
    2. Summation Over Repeated Indices: When an index appears twice in a term, it implies summation over that index.
    3. Free Indices: Indices that appear only once in the expression and are not summed over.
    """
    A = tf.constant([[[1, 2], [3, 4]],
                     [[5, 6], [7, 8]]], dtype=tf.float32)
    B = tf.constant([[[1, 0], [0, 1]],
                     [[1, 1], [1, 1]]], dtype=tf.float32)

    tf.einsum('ij,jk->ik', A, B)
    """
    The index j appears in both A and B, so we sum over all values of j.
    For each pair of indices (i, k), the value in the result tensor is obtained by summing 
    the product of elements across the shared dimension j.
    
    For i=0 and k=0:
    result[0,0]=∑j=0 (A[0,j]×B[j,0])
    
    For i=0 and k=1:
    result[0,1]=∑j=0(A[0,j]×B[j,1])
    """

    tf.einsum('abc,acd->abd', A, B)
    """
    Since a and c appears in both A and B, we sum over all values of a and c.
    For each pair of (b, d), the value in the result tensor is obtained by summing 
    the product of elements across the shared dimension (a, c).
    
    It simply result in batch wise multiplication
    """

    tf.einsum('bijc,bijd->bcd', A, A)      # Gram matrix
    """
    Since b, i, and j appear in both A and B, we sum over all values of b, i and j. For every pair of (b, c, d).
    Notice i and j are missing in output so we are doing summing over them. for every b,c and d

    Sum Over Spatial Dimensions:
    
    For b=0, c=0, d=0:
    Gram[0,0,0]= ∑ 0<=i<=3 ∑ 0<=j<=3 (input_tensor[0,i,j,0]×input_tensor[0,i,j,0])
    """

    # Similarly get the diagnol.
    diag = tf.einsum('ii->i', A)

    # Trace
    trace = tf.einsum('ii', A)

    # Dot product
    tf.einsum('i,i->', A, B)

    # Point 3: How to Resize image such that it maintains the aspect ratio

    img_path = ".."
    # tf.io.read_file(..) and then decode into the right format
    img = cv2.imread(img_path)[:, :, ::-1]  # [:, :, ::-1] converts BGR (opencv format...) into RGB
    # or img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    target_shape = 400   # The new desired height
    if target_shape is not None:  # resize section
        if isinstance(target_shape, int) and target_shape != -1:  # scalar -> implicitly setting the height
            current_height, current_width = img.shape[:2]   # We first retrieve the current spatial dimension.
            new_height = target_shape
            new_width = int(current_width * (new_height / current_height))  # We simply calculate the proportion of
            # width required for the aspect ratio to be maintained.
            img = cv2.resize(img, (new_width, new_height), interpolation=cv2.INTER_CUBIC)


