import numpy as np
import tensorflow as tf

def save_weights(graph, fpath):
    sess = tf.get_default_session()
    variables = graph.get_collection("variables")
    variable_names = [v.name for v in variables]
    kwargs = dict(zip(variable_names, sess.run(variables)))
    np.savez(fpath, **kwargs)

def load_weights_npy(weight_file, sess,G):
    weights = np.load(weight_file).item()
    if 'fc8' in weights:
        del weights['fc8']
        print('FC layer #8 will not be loaded')
    keys = sorted(weights.keys())

    with G.as_default():
        for key in keys:
            with tf.variable_scope(key, reuse=True):
                print(key, np.shape(weights[key]['weights']))
                sess.run(tf.get_variable('weights').assign(weights[key]['weights']))
                sess.run(tf.get_variable('biases').assign(weights[key]['biases']))
                print(tf.get_variable('weights').get_shape())

def load_weights_npz(graph, fpath):
    sess = tf.get_default_session()
    variables = graph.get_collection("variables")
    data = np.load(fpath)
    for v in variables:
        if v.name not in data:
            print("could not load data for variable='%s'" % v.name)
            continue
        print("assigning %s" % v.name)
        sess.run(v.assign(data[v.name]))
