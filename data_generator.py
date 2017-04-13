# Copyright (c) 2017 Yazabi Predictive Inc.

#################################### MIT License ####################################
#                                                                                   #
# Permission is hereby granted, free of charge, to any person obtaining a copy      #
# of this software and associated documentation files (the "Software"), to deal     #
# in the Software without restriction, including without limitation the rights      #
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell         #
# copies of the Software, and to permit persons to whom the Software is             #
# furnished to do so, subject to the following conditions:                          #
#                                                                                   #
# The above copyright notice and this permission notice shall be included in all    #
# copies or substantial portions of the Software.                                   #
#                                                                                   #
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR        #
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,          #
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE       #
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER            #
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,     #
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE     #
# SOFTWARE.                                                                         #
#                                                                                   #
#####################################################################################

# This is example code showing how to implement object-oriented TensorFlow.
# Note that normally each class would be in its own module (.py file). The code
# is heavily commented to allow you to follow along easily.

# Please report any bugs you find using @yazabi, or email us at contact@yazabi.com.

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

def create_samples(n_clusters, n_samples_per_cluster, n_features, embiggen_factor, seed):
    """Generates a set of samples in Gaussian distributions around a number
    of clusters equal to n_clusters. NOTE: dimensionality of the samples is defined
    by the num_features parameters (if num_features = 2, then the samples will be
    two-dimensional). We recommend sticking with n_features = 2 so that you can
    plot your results using matplotlib, but have made it a parameter here for
    those who want to experiment.

    :param n_clusters: The number of clusters around which the sample data are to be generated.
    :param n_samples_per_cluster: The number of samples generated around each cluster.
    :param n_features: The dimensionality of the samples (use n_features = 2 to plot the results).
    :param embiggen_factor: Defines the "spread" of the sample data around the clusters.
    :param seed: A random seed, for reproducibility.
    (see http://stackoverflow.com/questions/21494489/what-does-numpy-random-seed0-do#21494630)
    :return: The coordinates of the cluster centroids and randomly generated samples around them.
    """

    centroids = []
    samples = []

    tf.set_random_seed(seed)

    for _ in range(n_clusters):

        current_samples = tf.random_normal([n_samples_per_cluster, n_features], stddev=5.0)
        current_centroid = (tf.random_normal([1, n_features], stddev=5.0) * embiggen_factor) - (embiggen_factor/2)
        current_samples = current_samples + current_centroid

        samples.append(current_samples)
        centroids.append(current_centroid)

# Ops below only work in TensorFlow 1.0 and above. See https://www.tensorflow.org/api_docs/python/tf/concat
    samples = tf.concat(samples, 0, name='samples')
    centroids = tf.concat(centroids, 0, name='centroids')

    return centroids, samples

def choose_random_centroids(samples, n_clusters):
    """Selects and returns randomly chosen centroids from samples. Remember:
    samples is a tf.constant, not a numpy array, so you need to manipulate it
    using TensorFlow functions, and it won't have a value until it's run in a
    tf.Session().

    :param samples: A tf.constant of shape (number of samples) x (number of features).
    :param n_clusters: The number of clusters used to generate the data.
    :return: A tf.constant containing n_clusters randomly-selected samples
    from among `samples`. These random samples will be the initial centroid
    locations that your algorithm will use.
    """

    # choose n_clusters, or number of random points, to start the clustering
    n_samples = tf.shape(samples)[0]
    random_selection_indice = tf.random_shuffle(tf.range(0, n_samples))
    start = [0,]
    size = [n_clusters,]
    size[0] = n_clusters
    centroid_indices = tf.slice(random_selection_indice, start, size)
    initial_centroids = tf.gather(samples, centroid_indices)

    return initial_centroids

def assign_to_nearest(samples, centroids):
    """For each sample in `samples`, finds the index of the centroid in
    `centroids` that is closest to that sample.
    Hint: Use tf.expand_dims() to expand the dimensions of samples and centroids
    before you determine their separation distances. Note that tensor
    broadcasting in TensorFlow follows exactly the same rules as array
    broadcasting in numpy:
    https://docs.scipy.org/doc/numpy/user/basics.broadcasting.html
    So if you're not sure how two tensors are going to add together, try it out
    on the command line in numpy first. As always, message @yazabi if you get
    stuck!

    :param samples: A tf.constant of shape (number of samples) x (number of features).
    :param centroids: A tf.constant containing the locations of all current
    centroids.
    :return: The indices of the closest centroids to each sample.
    """

    new_vectors = tf.expand_dims(samples, 0)
    new_centroids = tf.expand_dims(centroids, 1)
    distances = tf.reduce_sum(
        tf.square(
            tf.subtract(new_vectors, new_centroids)
        ), 2
    )

    nearest_indices = tf.argmin(distances, 0)

    return nearest_indices

def update_centroids(samples, nearest_indices, n_clusters):
    """Determines the new centroid locations according to the K-means
    algorithm. Hint: you'll want to use tf.dynamic_partition() to
    define the samples whose positions will be averaged for each new
    centroid.

    :param samples: A tf.constant of shape (number of samples) x (number of features).
    :param nearest_indices: The indices of the closest centroids to each sample.
    :param n_clusters: The number of clusters.
    :return: A tf.constant giving the coordinates of the new centroids.
    """

    nearest_indices = tf.to_int32(nearest_indices)
    indice_partition = tf.dynamic_partition(samples, nearest_indices, n_clusters)
    updated_centroids = tf.concat([tf.expand_dims(tf.reduce_mean(partition, 0), 0) for partition in indice_partition], 0)

    return updated_centroids

# below is for debugging only.
def gen_samples(n_clusters, n_samples_per_cluster, n_features, embiggen_factor, seed):
    np.random.seed(seed)
    slices = []
    centroids = []
    # Create samples for each cluster
    for i in range(n_clusters):
        samples = tf.random_normal((n_samples_per_cluster, n_features),
                               mean=0.0, stddev=5.0, dtype=tf.float32, seed=seed, name="cluster_{}".format(i))
        current_centroid = (np.random.random((1, n_features)) * embiggen_factor) - (embiggen_factor/2)
        centroids.append(current_centroid)
        samples += current_centroid
        slices.append(samples)
    # Create a big "samples" dataset
        samples = tf.concat(slices, 0, name='samples')
        centroids = tf.concat(centroids, 0, name='centroids')
        centroids = tf.cast(centroids, tf.float32)
        return centroids, samples

