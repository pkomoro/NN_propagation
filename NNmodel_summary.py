import tensorflow as tf

import lightprop.propagation.methods as prop
from lightprop.propagation.params import PropagationParams
from lightprop.calculations import get_lens_distribution
import numpy as np

# Prepare propagator
params = PropagationParams.get_example_propagation_data()
propagator = prop.MultiparameterNNPropagation_FFTConv()

initial_weights = np.mod(get_lens_distribution(params),2*np.pi)

# Extract network with dimensions and trainable weights
model = propagator.build_model(params.matrix_size, initial_weights)
# print(model._get_trainable_state())

# Print model summary
model.summary()


# Save network graph
dot_img_file = "outs/model.png"
tf.keras.utils.plot_model(model, to_file=dot_img_file, show_shapes=True, dpi=1000)
