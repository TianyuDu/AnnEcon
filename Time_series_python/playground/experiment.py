import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import math

# 1-d convolutional layer
def conv1d(X, num_filters=8, filter_width=3, stride=1, padding='SAME'):
    # helper function for a 1D convolutional filter
    # initalize filter
    window_size = int(X.get_shape()[1])
    num_sensors = int(X.get_shape()[2])
    stddev = 1
    f = tf.Variable(tf.truncated_normal((filter_width,num_sensors,num_filters),stddev=.2),trainable=True,name='conv1d_filter')
    # initialize bias
    b = tf.Variable(0.0,name='conv1d_bias')
    conv = tf.nn.conv1d(value=X,filters=f,stride=stride,padding=padding,name='conv1d_op')
    return tf.add(conv,b)

# print out graph structure
def print_graph():
    # prints the graph operations out
    with tf.Session() as sess:
        op = sess.graph.get_operations()
    for o in op:
        print o.outputs

# container to hold cnnrnn model structure
class cnnrnn_model:
    def __init__(self,time_steps,window_size,num_sensors,filters,filter_size,rnn_nodes):
        ###### model creation #############################################################
        # placeholders
        self.X = tf.placeholder(tf.float32,[None,time_steps,window_size,num_sensors],name='X')
        self.Y = tf.placeholder(tf.float32,[None,time_steps,1],name='Y')

        # create the convolutional layers for each CNN per time step
        m = []
        for i in range(0,time_steps):
            # batch, time_step, window_size, num_sensors
            m1 = conv1d(self.X[:,i,:,:], num_filters=filters*1, filter_width=filter_size, stride=1, padding='SAME')
            m1 = tf.nn.relu(m1,name='relu1d')
            m1 = tf.nn.pool(m1, window_shape=(4,), pooling_type='MAX', padding='SAME', strides=(4,), name='pool1d')
            m1 = conv1d(m1, num_filters=filters*1, filter_width=filter_size, stride=1, padding='SAME')
            m1 = tf.nn.relu(m1,name='relu1d')
            m1 = tf.nn.pool(m1, window_shape=(4,), pooling_type='MAX', padding='SAME', strides=(4,), name='pool1d')
            m1 = conv1d(m1, num_filters=filters*1, filter_width=filter_size, stride=1, padding='SAME')
            m1 = tf.nn.relu(m1,name='relu1d')
            m1 = tf.nn.pool(m1, window_shape=(2,), pooling_type='MAX', padding='SAME', strides=(2,), name='pool1d')
            m1 = conv1d(m1, num_filters=filters*1, filter_width=1, stride=1, padding='SAME')
            m1 = tf.nn.relu(m1,name='relu1d')
            m1 = tf.nn.pool(m1, window_shape=(2,), pooling_type='MAX', padding='SAME', strides=(2,), name='pool1d')
            sh1 = int(m1.get_shape()[1])
            sh2 = int(m1.get_shape()[2])
            m1 = tf.reshape(m1, [-1,1,sh1*sh2])
            m.append(m1)
            
        c = tf.concat(m,1)

        basic_cell = tf.nn.rnn_cell.BasicRNNCell(num_units=rnn_nodes)
        model, states = tf.nn.dynamic_rnn(cell=basic_cell, inputs=c, dtype=tf.float32, time_major=False)

        self.model = tf.layers.dense(model,units=1,activation=None)

        self.loss = tf.losses.mean_squared_error(self.Y, self.model)

        optimizer = tf.train.AdamOptimizer(1e-3)
        self.training_op = optimizer.minimize(self.loss) 
        self.init = tf.global_variables_initializer()
        ###### end model creation #############################################################

# some simulated data to play with
def create_test_data(batch_size, time_steps, window_size, num_sensors):
    # create fake training data for testing neural net
    x = np.zeros((batch_size,time_steps,window_size,num_sensors))
    y = np.zeros((batch_size,time_steps,1)) # these are the outputs of the RNN + dense layer
    num_examples = batch_size + time_steps
    xe = np.zeros((num_examples,window_size,num_sensors))
    ye = np.zeros((num_examples,1))
    # normal case (no fault)
    for e in range(0,num_examples/2): # each example
        wn = 1
        d = 0.73
        c2 = -wn*wn
        c1 = -2*d*wn
        c3 = 1
        x1 = 0
        x2 = 0
        for s in range(0,window_size):  # each sample
            x1 = x1 + 0.4*x2
            x2 = x2 + 0.4*(c1*x2 +c2*x1 + c3)
            xe[e,s,0] = -x1*c2 + np.random.randn()*0.1
            xe[e,s,1] = x2 + np.random.randn()*0.1
            ye[e,0] = 1.0
    # fault case (damping coefficient changing)
    for e in range(num_examples/2,num_examples):
        i = e-num_examples/2
        wn = 1
        d = 0.72 - 0.3*float(i+1)/(num_examples/2)
        c2 = -wn*wn
        c1 = -2*d*wn
        c3 = 1
        x1 = 0
        x2 = 0
        for s in range(0,window_size):
            x1 = x1 + 0.4*x2
            x2 = x2 + 0.4*(c1*x2 +c2*x1 + c3)
            xe[e,s,0] = -x1*c2 + np.random.randn()*0.1
            xe[e,s,1] = x2 + np.random.randn()*0.1
            ye[e,0] = math.exp(-0.1*i)
    # reorganize data into timesteps
    for b in range(0,batch_size):
        for t in range(0,time_steps):    
            x[b,t,:,:] = xe[b + t,:,:]
            y[b,t,:] = ye[b + t,:]
    return x,y

###### model parameters ###########################################################
time_steps = 5
window_size = 64
num_sensors = 2 
filters = 6
filter_size= 3
rnn_nodes= 8

###### training parameters ########################################################
batch_size = 64
n_epochs = 501

###### create test data ###########################################################
trainX, trainY = create_test_data(batch_size, time_steps, window_size, num_sensors)

###### model creation #############################################################
model = cnnrnn_model(time_steps,window_size,num_sensors,filters,filter_size,rnn_nodes)

###### saver object to save and restore model variables ###########################
saver = tf.train.Saver()

###### model training #############################################################
with tf.Session() as sess:
    sess.run(model.init)       
    for e in range(0,n_epochs):
        sess.run(model.training_op, feed_dict={model.X: trainX, model.Y: trainY})
        loss_out = sess.run(model.loss, feed_dict={model.X: trainX, model.Y: trainY})
        if (e+1) % 100 == 1:
            print 'epoch = ' + str(e+1) + '/' + str(n_epochs) + ', loss = ' + str(loss_out)
        if (e+1) % 5000 == 1:
            print 'epoch = ' + str(e+1) + '/' + str(n_epochs) + ', loss = ' + str(loss_out)
    saver.save(sess,'/tmp/test-model')
    result = sess.run(model.model, feed_dict={model.X: trainX, model.Y: trainY})
    print result.T
###### end model training #########################################################

###### example restoring model ####################################################
tf.reset_default_graph()
with tf.Session() as sess:
    new_saver = tf.train.import_meta_graph('/tmp/test-model.meta')
    new_saver.restore(sess,tf.train.latest_checkpoint('/tmp/'))
    graph = tf.get_default_graph()
    X = graph.get_tensor_by_name("X:0")
    Y = graph.get_tensor_by_name("Y:0")
    model = graph.get_tensor_by_name('dense/BiasAdd:0')
    result = sess.run(model, feed_dict={X: trainX, Y: trainY})
    print result.T