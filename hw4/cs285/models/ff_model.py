import tensorflow as tf

from .base_model import BaseModel
from cs285.infrastructure.utils import normalize, unnormalize
from cs285.infrastructure.tf_utils import build_mlp


class FFModel(BaseModel):

    def __init__(self, sess, ac_dim, ob_dim, n_layers, size, learning_rate=0.001, scope='dyn_model'):
        super(FFModel, self).__init__()

        # init vars
        # self.env = env
        self.sess = sess
        self.ac_dim = ac_dim
        self.ob_dim = ob_dim
        self.n_layers = n_layers
        self.size = size
        self.learning_rate = learning_rate
        self.scope = scope

        # build TF graph
        with tf.variable_scope(self.scope, reuse=tf.AUTO_REUSE):
            self.build_graph()
        self.define_train_op()

    #############################

    def build_graph(self):
        self.define_placeholders()
        self.define_forward_pass()

    def define_placeholders(self):

        self.obs_pl = tf.placeholder(shape=[None, self.ob_dim], name="ob", dtype=tf.float32)
        self.acs_pl = tf.placeholder(shape=[None, self.ac_dim], name="ac", dtype=tf.float32)
        self.delta_labels = tf.placeholder(shape=[None, self.ob_dim], name="labels", dtype=tf.float32)

        self.obs_mean_pl = tf.placeholder(shape=[self.ob_dim], name="obs_mean", dtype=tf.float32)
        self.obs_std_pl = tf.placeholder(shape=[self.ob_dim], name="obs_std", dtype=tf.float32)
        self.acs_mean_pl = tf.placeholder(shape=[self.ac_dim], name="acs_mean", dtype=tf.float32)
        self.acs_std_pl = tf.placeholder(shape=[self.ac_dim], name="acs_std", dtype=tf.float32)
        self.delta_mean_pl = tf.placeholder(shape=[self.ob_dim], name="delta_mean", dtype=tf.float32)
        self.delta_std_pl = tf.placeholder(shape=[self.ob_dim], name="delta_std", dtype=tf.float32)

    def define_forward_pass(self):
        # normalize input data to mean 0, std 1
        obs_unnormalized = self.obs_pl
        acs_unnormalized = self.acs_pl
        # Hint: Consider using the normalize function defined in infrastructure.utils for the following two lines
        obs_normalized = # TODO(Q1) Define obs_normalized using obs_unnormalized,and self.obs_mean_pl and self.obs_std_pl
        acs_normalized = # TODO(Q2) Define acs_normalized using acs_unnormalized and self.acs_mean_pl and self.acs_std_pl

        # predicted change in obs
        concatenated_input = tf.concat([obs_normalized, acs_normalized], axis=1)
        # Hint: Note that the prefix delta is used in the variable below to denote changes in state, i.e. (s'-s)
        self.delta_pred_normalized = # TODO(Q1) Use the build_mlp function and the concatenated_input above to define a neural network that predicts unnormalized delta states (i.e. change in state)
        self.delta_pred_unnormalized = # TODO(Q1) Unnormalize the the delta_pred above using the unnormalize function, and self.delta_mean_pl and self.delta_std_pl
        self.next_obs_pred = # TODO(Q1) Predict next observation using current observation and delta prediction (not that next_obs here is unnormalized)

    def define_train_op(self):

        # normalize the labels
        self.delta_labels_normalized =  # TODO(Q1) Define a normalized version of delta_labels using self.delta_labels (which are unnormalized), and self.delta_mean_pl and self.delta_std_pl

        # compared predicted deltas to labels (both should be normalized)
        self.loss = # TODO(Q1) Define a loss function that takes as input normalized versions of predicted change in state and ground truth change in state
        self.train_op = # TODO(Q1) Define a train_op to minimize the loss defined above. Adam optimizer will work well.

    #############################

    def get_prediction(self, obs, acs, data_statistics):
        if len(obs.shape)>1:
            observations = obs
            actions = acs
        else:
            observations = obs[None]
            actions = acs [None]
        return # TODO(Q1) Run model prediction on the given batch of data

    def update(self, observations, actions, next_observations, data_statistics):
        # train the model
        _, loss = # TODO(Q1) Run the defined train_op here, and also return the loss being optimized (on this batch of data)
        return loss