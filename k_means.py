# Yazabi Grads
# TensorFlow Module 2
# Completed by Kamil Krawczyk

from data_generator import *

def plot_results(all_data, centroids, n_samples_per_cluster):
    # set up packages
    import numpy as np
    import matplotlib.pyplot as plt

    colour = plt.cm.rainbow(np.linspace(0,1,len(centroids)))
    for i, centroid in enumerate(centroids):
        samples = all_data[i*n_samples_per_cluster:(i+1)*n_samples_per_cluster]
        plt.scatter(samples[:,0], samples[:,1], c = colour[i])
        plt.plot(centroid[0], centroid[1], markersize = 35, marker = 'x', color = 'k', mew = 10)
        plt.plot(centroid[0], centroid[1], markersize = 30, marker = 'x', color = 'm', mew = 5)
    plt.show()

def run_k_means(n_clusters, samples_per_cluster, n_features, embiggen_factor, seed):
    # set up packages
    import tensorflow as tf

    # create data, and perform K means clustering
    data_centroids, samples = create_samples(n_clusters, samples_per_cluster, n_features, embiggen_factor, seed)
    initial_k_centroids = choose_random_centroids(samples, n_clusters)
    nearest_indices = assign_to_nearest(samples, initial_k_centroids)
    updated_centroids = update_centroids(samples, nearest_indices, n_clusters)

    # perform TF session
    model = tf.global_variables_initializer()
    with tf.Session() as session:
        data_values = session.run(samples)
        updated_centroid_value = session.run(updated_centroids)
        print(updated_centroid_value)
        print(data_values)

    # plot results
    plot_results(data_values, updated_centroid_value, samples_per_cluster)

    # return nothing
    return None

def main():

    # define parameters
    n_clusters = 3
    samples_per_cluster = 200
    n_features = 2
    seed = 700
    embiggen_factor = 70

    # run the function
    run_k_means(n_clusters, samples_per_cluster, n_features, embiggen_factor, seed)

# run program
if __name__ == "__main__":
    main()