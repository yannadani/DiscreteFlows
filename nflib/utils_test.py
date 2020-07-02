import torch
import tensorflow as tf
import edward2 as ed
import numpy as np

from utils import one_hot, one_hot_argmax, multiplicative_inverse, one_hot_add, one_hot_minus, one_hot_multiply

#np.random.seed(10)
rm = np.random.randint(100, size=(100,10,10))
shift = np.random.randint(100, size=(100,10,10))

one_hot_torch = one_hot(torch.from_numpy(rm), 101)
one_hot_tf = tf.one_hot(tf.convert_to_tensor(rm), depth = 101)

shift_oh_t = one_hot(torch.from_numpy(shift), 101)
shift_oh_tf = tf.one_hot(tf.convert_to_tensor(shift), depth = 101)

assert np.allclose(one_hot_torch, one_hot_tf)

############################################################


oha_torch = one_hot_argmax(one_hot_torch, 0.1)
oha_tf = ed.layers.utils.one_hot_argmax(one_hot_tf, 0.1)

assert np.allclose(oha_torch, oha_tf)

#############################################################

ohm_torch = one_hot_minus(one_hot_torch, shift_oh_t)
ohm_tf = ed.layers.utils.one_hot_minus(one_hot_tf, shift_oh_tf)

assert np.allclose(ohm_torch, ohm_tf)
#############################################################
ohml_torch = one_hot_multiply(one_hot_torch, shift_oh_t)
ohml_tf = ed.layers.utils.one_hot_multiply(one_hot_tf, shift_oh_tf)

assert np.allclose(ohml_torch, ohml_tf)

###############################################################

ohad_torch = one_hot_add(one_hot_torch, shift_oh_t)
ohad_tf = ed.layers.utils.one_hot_add(one_hot_tf, shift_oh_tf)
assert np.allclose(ohad_torch, ohad_tf)

###############################################################
mi_torch = multiplicative_inverse(one_hot_torch, 3)
mi_tf = ed.layers.utils.multiplicative_inverse(one_hot_tf, 3)

assert np.allclose(mi_torch, mi_tf)
