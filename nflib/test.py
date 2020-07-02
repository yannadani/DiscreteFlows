import torch
import tensorflow as tf
import numpy as np
from utils import one_hot
def one_hot_add(inputs, shift):
    """Performs (inputs + shift) % vocab_size in the one-hot space.
    Args:
        inputs: Tensor of shape `[..., vocab_size]`. Typically a soft/hard one-hot
        Tensor.
        shift: Tensor of shape `[..., vocab_size]`. Typically a soft/hard one-hot
        Tensor specifying how much to shift the corresponding one-hot vector in
        inputs. Soft values perform a "weighted shift": for example,
        shift=[0.2, 0.3, 0.5] performs a linear combination of 0.2 * shifting by
        zero; 0.3 * shifting by one; and 0.5 * shifting by two.
    Returns:
        Tensor of same shape and dtype as inputs.
    """
    inputs = torch.stack((inputs, torch.zeros_like(inputs)), dim = -1)
    shift = torch.stack((shift, torch.zeros_like(shift)), dim = -1)
    inputs_fft = torch.fft(inputs, 1) #ignore last and first dimension to do batched fft
    shift_fft = torch.fft(shift, 1)
    result_fft_real = inputs_fft[...,0]*shift_fft[...,0] - inputs_fft[...,1]*shift_fft[...,1]
    result_fft_imag = inputs_fft[...,0]*shift_fft[...,1] + inputs_fft[...,1]*shift_fft[...,0]
    result_fft = torch.stack((result_fft_real,result_fft_imag), dim = -1)
    return torch.ifft(result_fft, 1)[...,0], result_fft, inputs_fft, shift_fft #return only the real part

def one_hot_add_tf(inputs, shift):
  """Performs (inputs + shift) % vocab_size in the one-hot space.
  Args:
    inputs: Tensor of shape `[..., vocab_size]`. Typically a soft/hard one-hot
      Tensor.
    shift: Tensor of shape `[..., vocab_size]`. Typically a soft/hard one-hot
      Tensor specifying how much to shift the corresponding one-hot vector in
      inputs. Soft values perform a "weighted shift": for example,
      shift=[0.2, 0.3, 0.5] performs a linear combination of 0.2 * shifting by
      zero; 0.3 * shifting by one; and 0.5 * shifting by two.
  Returns:
    Tensor of same shape and dtype as inputs.
  """
  # Compute circular 1-D convolution with shift as the kernel.
  inputs = tf.cast(inputs, tf.complex64)
  shift = tf.cast(shift, tf.complex64)
  inputs_fft = tf.signal.fft(inputs)
  shift_fft = tf.signal.fft(shift)
  result_fft = inputs_fft * shift_fft
  return tf.math.real(tf.signal.ifft(result_fft)), result_fft, inputs_fft, shift_fft

if __name__ == '__main__':
    rm = np.random.randint(100, size=(100,10,10))
    shift = np.random.randint(100, size=(100,10,10))

    one_hot_torch = one_hot(torch.from_numpy(rm), 101)
    one_hot_tf = tf.one_hot(tf.convert_to_tensor(rm), depth = 101)

    shift_oh_t = one_hot(torch.from_numpy(shift), 101)
    shift_oh_tf = tf.one_hot(tf.convert_to_tensor(shift), depth = 101)

    assert np.allclose(shift_oh_t, shift_oh_tf)
    assert np.allclose(one_hot_torch, one_hot_tf)
    
    res_1, res_fft1, ip_fft1, sh_fft1 = one_hot_add(one_hot_torch, shift_oh_t)
    res_2, res_fft2, ip_fft2, sh_fft2 = one_hot_add_tf(one_hot_tf, shift_oh_tf)

    assert np.allclose(ip_fft1[...,0], tf.math.real(ip_fft2), rtol = 0.001)
    #assert np.allclose(ip_fft1[...,1], tf.math.imag(ip_fft2), rtol = 0.1)
    assert np.allclose(sh_fft1[...,0], tf.math.real(sh_fft2), rtol = 0.001)
    #assert np.allclose(sh_fft1[...,1], tf.math.imag(sh_fft2), rtol = 0.001)
    assert np.allclose(res_fft1[...,0], tf.math.real(res_fft2), rtol = 0.001)
    #assert np.allclose(res_fft1[...,1], tf.math.imag(res_fft2), rtol = 0.001)
    assert np.array_equal(torch.argmax(res_1, dim = -1), tf.argmax(res_2, axis = -1))