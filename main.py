# TensorFlow skill testing section
# Kamil Krawczyk 2017 - for Yazabi Grads

import tensorflow as tf
import numpy as np

import tensorflow as tf
import numpy as np

from data_generator import *
from k_means import plot_results

n_features = 2
n_clusters = 3
n_samples_per_cluster = 500
seed = 700
embiggen_factor = 70

np.random.seed(seed)

centroids, samples = create_samples(n_clusters, n_samples_per_cluster, n_features, embiggen_factor, seed)

model = tf.global_variables_initializer()
with tf.Session() as session:
    sample_values = session.run(samples)
    centroid_values = session.run(centroids)

plot_results(sample_values, centroid_values, n_samples_per_cluster)
