
import tensorflow as tf
import numpy as np



class Network(object):
    """
    Input to the network is the state, output is the action
    under a deterministic policy.
    The output layer activation is a tanh to keep the action
    between -2 and 2
    """

    def __init__(self, sess, state_size, action_dim, learning_rate, tau, device):
        self.sess = sess
        self.a_dim = action_dim
        self.learning_rate = learning_rate
        self.tau = tau
        self.currentState = -1.
        self.device = device
        self.state_size = state_size
        
        # Q network
        self.inputs, self.out, self.saver = self.create_q_network('q')
        self.network_params = tf.trainable_variables()

        # Target Network
        self.target_inputs, self.target_out, self.target_saver = self.create_q_network('q_target')

        self.target_network_params = tf.trainable_variables()[len(self.network_params):]

        # Op for periodically updating target network with online network
        # weights
        self.update_target_network_params = [self.target_network_params[i].assign(self.network_params[i]) for i in range(len(self.target_network_params))]
        #self.update_target_network_params = tf.assign(self.target_network_params, self.network_params)

        with tf.device(self.device):

            self.target_q_t = tf.placeholder(tf.float32, [None, self.a_dim], name='target_q')
            self.action = tf.placeholder(tf.int32, [None, 1])
            action_one_hot = tf.one_hot(self.action, self.a_dim, 1.0, 0.0, name='action_one_hot')
            q_acted = tf.reduce_sum(self.out * action_one_hot, reduction_indices=1, name='q_acted')
            self.delta = self.target_q_t - q_acted
            self.loss = tf.reduce_mean(self.clipped_error(self.delta), name='loss')
            #self.optimizer = tf.train.RMSPropOptimizer(self.learning_rate,momentum=0.95, epsilon=0.01)#, 0.99, 0.0, 1e-6)
            #self.l2_loss = 0.01*(tf.nn.l2_loss(self.weights1)) +  0.01*(tf.nn.l2_loss(self.weights2)) + 0.01*(tf.nn.l2_loss(self.weights3))


            self.optimize = tf.train.AdamOptimizer(self.learning_rate).minimize(self.loss)
            #self.optimize = self.optimizer.minimize(self.loss)
            '''
            self.predicted_q_value = tf.placeholder(tf.float32, [None, self.a_dim])
            # Define loss and optimization Op
            self.loss = tf.reduce_mean(tf.square(tf.subtract(self.predicted_q_value,self.out)))
            self.optimize = tf.train.AdamOptimizer(self.learning_rate).minimize(self.loss)
            '''

        self.num_trainable_vars = len(self.network_params) + len(self.target_network_params)

    def create_q_network(self, scope):

        with tf.device(self.device):
            with tf.variable_scope(scope): 
                
                stateInput = tf.placeholder(tf.float32, shape=[None,self.state_size])
                # fully connected layer
                # stateInput = tf.to_float(stateInput)
                fc = tf.contrib.layers.fully_connected(stateInput, 100)
                fc2 = tf.contrib.layers.fully_connected(fc, 100)
                out = tf.contrib.layers.fully_connected(fc2, self.a_dim,activation_fn=None)
                
               
        saver = tf.train.Saver()
        return stateInput, out, saver

        
    def train(self, actions, target_q_t, inputs):
        with tf.device(self.device):
            self.sess.run(self.optimize, feed_dict={
                self.action: actions,
                self.target_q_t: target_q_t,
                self.inputs: inputs
            })

    def predict(self, inputs):
        with tf.device(self.device):
            return self.sess.run(self.out, feed_dict={
                self.inputs: inputs
            })

    def predict_target(self, inputs):
        return self.sess.run(self.target_out, feed_dict={
            self.target_inputs: inputs
        })

    
    def update_target_network(self):
        with tf.device(self.device):
            self.sess.run(self.update_target_network_params)

    def get_num_trainable_vars(self):
        return self.num_trainable_vars
    
    def save(self):
        self.saver.save(self.sess,'./model.ckpt')
        self.target_saver.save(self.sess,'./model_target.ckpt')
        #saver.save(self.sess,'actor_model.ckpt')
        print("Model saved in file: model")

    
    def recover(self):
        self.saver.restore(self.sess,'./model.ckpt')
        self.target_saver.restore(self.sess,'./model_target.ckpt')
        #saver.restore(self.sess,'critic_model.ckpt')
    
    def conv2d(self,x, W, stride):
        with tf.device(self.device):
            return tf.nn.conv2d(x, W, strides = [1, stride, stride, 1], padding = "VALID")

    def clipped_error(self,x):
      # Huber loss
      try:
        return tf.select(tf.abs(x) < 1.0, 0.5 * tf.square(x), tf.abs(x) - 0.5)
      except:
        return tf.where(tf.abs(x) < 1.0, 0.5 * tf.square(x), tf.abs(x) - 0.5)