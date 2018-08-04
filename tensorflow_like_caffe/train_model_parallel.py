import sys
import os
import os.path as pth
import time
import numpy as np
import tensorflow as tf
import json
import vgg
import layers as L
import yaml
import tools
import argparse
import train_data_generator
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import data_flow_ops

# =====================================
# Training configuration default params
# =====================================
config = {}

# =========================
# customize your model here
# =========================
def build_model(input_data_tensor, input_label_tensor, num_classes,scope):
    logits,_ = vgg.intfVGG(input_data_tensor, n_classes=num_classes)
    loss = vgg.loss(logits, input_label_tensor)
    # Assemble all of the losses for the current tower only.
    # scope: (Optional.) If supplied, the resulting list is filtered to include only items whose name attribute matches using re.match.
    losses_one_tower = tf.get_collection('losses', scope)
    # Calculate the total loss for the current tower.
    total_loss_one_tower = tf.add_n(losses_one_tower, name='total_loss')
    # error_top5 = L.topK_error(probs, input_label_tensor, K=5)
    # error_top1 = L.topK_error(probs, input_label_tensor, K=1)

    # you must return a dictionary with at least the "loss" as a key
    return dict(loss=total_loss_one_tower)
                # logits=logits,
                # error_top5=error_top5,
                # error_top1=error_top1)


# =================================
#  generice multi-gpu training code
# =================================
def train():
    checkpoint_dir = config["checkpoint_dir"]
    learning_rate = config['learning_rate']
    image_size = config['image_size']
    batch_size = config['batch_size']
    num_gpus = config['num_gpus']
    num_epochs = config['num_epochs']
    pretrained_weights = config["pretrained_weights"]
    checkpoint_iter = config["checkpoint_iter"]
    experiment_dir = config['experiment_dir']
    data_folder = config['data_folder']
    split_ratio = config['split_ratio']
    min_nrof_images_per_class = config['min_nrof_images_per_class']
    nrof_preprocess_threads = config['nrof_preprocess_threads']
    learning_rate_decay_epochs = config['learning_rate_decay_epochs']
    learning_rate_decay_factor = config['learning_rate_decay_factor']
    opt=config['opt']
    train_log_fpath = pth.join(experiment_dir, 'train.log')

    # ====================
    # get the data set, note that you only specify the folder that contains subfolders
    # which named by label_name is OK
    # ====================
    dataset = train_data_generator.get_dataset(data_folder)
    train_set, val_set=train_data_generator.split_dataset(dataset, split_ratio, min_nrof_images_per_class, mode='SPLIT_IMAGES')
    nrof_classes = len(train_set)
    print 'My classes is: ', nrof_classes, split_ratio
    raw_input("------------")
    # in train_set, we store {label:[many photos]}, now change to [label...]:[photo...]
    image_list, label_list = train_data_generator.get_image_paths_and_labels(train_set)
    # val_image_list, val_label_list = train_data_generator.get_image_paths_and_labels(val_set)
    
    num_samples_per_epoch = len(image_list)
    steps_per_epoch = num_samples_per_epoch // (batch_size * num_gpus)
    num_steps = steps_per_epoch * num_epochs
    print 'My steps_per_epoch is: ', steps_per_epoch
    raw_input("------------")
    # =====================
    # define training graph
    # =====================
    G = tf.Graph()
    with G.as_default(), tf.device('/cpu:0'):
        # at first we define a index queue 
        # for reading (photo[index...index],label[index...index]
        # note that we should shuffle
        # may not worry to the capacity, queue will block either full or empty
        labels = ops.convert_to_tensor(label_list, dtype=tf.int32)
        range_size = array_ops.shape(labels)[0]
        index_queue = tf.train.range_input_producer(range_size, num_epochs=None,
                             shuffle=True, seed=None, capacity=num_samples_per_epoch)
        index_dequeue_op = index_queue.dequeue_many(num_samples_per_epoch, 'index_dequeue')

        # then create a queue for [image_path,label]
        input_queue = data_flow_ops.FIFOQueue(capacity=num_samples_per_epoch*2,
                                    dtypes=[tf.string, tf.int32, tf.int32],
                                    shapes=[(1,), (1,), (1,)],
                                    shared_name=None, name=None)
        image_paths_placeholder = tf.placeholder(tf.string, shape=(None,1), name='image_paths')
        labels_placeholder = tf.placeholder(tf.int32, shape=(None,1), name='labels')
        control_placeholder = tf.placeholder(tf.int32, shape=(None,1), name='control_value')
        enqueue_op = input_queue.enqueue_many([image_paths_placeholder, labels_placeholder, control_placeholder], name='enqueue_op')
        data, labels = train_data_generator.create_input_pipeline(input_queue, (image_size,image_size), nrof_preprocess_threads, batch_size*num_gpus)



        # we split the large batch into sub-batches to be distributed onto each gpu
        split_data = tf.split(data, num_gpus, 0)
        split_labels = tf.split(labels, num_gpus, 0)

        
        learning_rate_placeholder = tf.placeholder(tf.float32, name='learning_rate')
        global_step= tf.placeholder(tf.int32, name='global_step')
        learning_rate_op = tf.train.exponential_decay(learning_rate_placeholder, global_step,
            learning_rate_decay_epochs*steps_per_epoch, learning_rate_decay_factor, staircase=True)
        
        
        if opt=='ADAGRAD':
            optimizer = tf.train.AdagradOptimizer(learning_rate_op)
        elif opt=='ADADELTA':
            optimizer = tf.train.AdadeltaOptimizer(learning_rate_op, rho=0.9, epsilon=1e-6)
        elif opt=='ADAM':
            optimizer = tf.train.AdamOptimizer(learning_rate_op, beta1=0.9, beta2=0.999, epsilon=0.1)
        elif opt=='RMSPROP':
            optimizer = tf.train.RMSPropOptimizer(learning_rate_op, decay=0.9, momentum=0.9, epsilon=1.0)
        else:
            optimizer = tf.train.MomentumOptimizer(learning_rate_op, 0.9, use_nesterov=True)

        # setup one model replica per gpu to compute loss and gradient
        replica_grads = []
        with tf.variable_scope(tf.get_variable_scope()):
            for i in range(num_gpus):
                with tf.name_scope('tower_%d' % i), tf.device('/gpu:%d' % i):
                    model = build_model(split_data[i], split_labels[i],nrof_classes,'tower_%d' % i)
                    loss = model["loss"]
                    tf.get_variable_scope().reuse_variables()
                    grads = optimizer.compute_gradients(loss)
                    replica_grads.append(grads)

        # We must calculate the mean of each gradient.
        average_grad = L.average_gradients(replica_grads)
        grad_step = optimizer.apply_gradients(average_grad)
        train_step = tf.group(grad_step)
        init = tf.global_variables_initializer()
        init2 = tf.local_variables_initializer()

    # ==================
    # run training graph
    # ==================
    config_proto = tf.ConfigProto(allow_soft_placement=True)
    sess = tf.Session(graph=G, config=config_proto)
    sess.run([init,init2])
    summary_writer = tf.summary.FileWriter('log', sess.graph)
    coord = tf.train.Coordinator()
    tf.train.start_queue_runners(coord=coord, sess=sess)

    with sess.as_default():
        if pretrained_weights:
            print("-- loading weights from %s" % pretrained_weights)
            if 'npz' in pretrained_weights:
                tools.load_weights_npz(G, pretrained_weights)
            elif 'npy' in pretrained_weights:
                tools.load_weights_npy(pretrained_weights,sess,G)
            else:
                print 'unknow file, skip load weights'

        for step in range(num_steps):
            if step%steps_per_epoch == 0:
                # after a epoch, fetch datas, THIS pipeline performance is not so satisfying for me
                # MAY BE WILL replace late :(
                print('Loading data ...',num_steps)
                index_epoch = sess.run(index_dequeue_op)
                label_epoch = np.array(label_list)[index_epoch]
                image_epoch = np.array(image_list)[index_epoch]
                # Enqueue one epoch of image paths and labels
                labels_array = np.expand_dims(np.array(label_epoch),1)
                image_paths_array = np.expand_dims(np.array(image_epoch),1)
                control_value = train_data_generator.RANDOM_ROTATE * config['random_rotate'] + train_data_generator.RANDOM_CROP * config['random_crop'] + train_data_generator.RANDOM_FLIP * config['random_flip'] + train_data_generator.FIXED_STANDARDIZATION * config['use_fixed_image_standardization']
                control_array = np.ones_like(labels_array) * control_value
                # after enqueue_op, input is send to tower
                sess.run(enqueue_op, {image_paths_placeholder: image_paths_array, labels_placeholder: labels_array, control_placeholder: control_array})
                print('reading end')

            inputs = {learning_rate_placeholder: learning_rate, global_step: step}
            results = sess.run([train_step, loss, learning_rate_op], inputs)
            print("step:%s loss:%s lr:%s" % (step, results[1], results[2]))

            if (step%(steps_per_epoch*checkpoint_iter) == 0 and step != 0) or (step + 1 == num_steps):
                print("-- saving check point")
                tools.save_weights(G, pth.join(checkpoint_dir, "weights_%s" % step))



def main(argv=None):
    num_gpus = config['num_gpus']
    batch_size = config['batch_size']
    checkpoint_dir = config["checkpoint_dir"]
    experiment_dir = config["experiment_dir"]

    # setup experiment and checkpoint directories
    if not pth.exists(experiment_dir):
        os.makedirs(experiment_dir)
    if not pth.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)

    train()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_file', help='YAML formatted config file', type=str, default='experiment.yaml')
    args = parser.parse_args()
    with open(args.config_file) as fp:
        config.update(yaml.load(fp))

    print "Experiment config"
    print "------------------"
    print json.dumps(config, indent=4)
    print "------------------"
    main()
