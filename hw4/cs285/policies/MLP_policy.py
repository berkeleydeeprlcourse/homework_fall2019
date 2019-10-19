import numpy as np
import tensorflow as tf
from .base_policy import BasePolicy
from cs285.infrastructure.tf_utils import build_mlp
import tensorflow_probability as tfp

class MLPPolicy(BasePolicy):

    def __init__(self,
        sess,
        ac_dim,
        ob_dim,
        n_layers,
        size,
        learning_rate=1e-4,
        training=True,
        policy_scope='policy_vars',
        discrete=False, # unused for now
        nn_baseline=False, # unused for now
        **kwargs):
        super().__init__(**kwargs)

        # init vars
        self.sess = sess
        self.ac_dim = ac_dim
        self.ob_dim = ob_dim
        self.n_layers = n_layers
        self.size = size
        self.discrete = discrete
        self.learning_rate = learning_rate
        self.training = training
        self.nn_baseline = nn_baseline

        # build TF graph
        with tf.variable_scope(policy_scope, reuse=tf.AUTO_REUSE):
            self.build_graph()

        # saver for policy variables that are not related to training
        self.policy_vars = [v for v in tf.all_variables() if policy_scope in v.name and 'train' not in v.name]
        self.policy_saver = tf.train.Saver(self.policy_vars, max_to_keep=None)

    ##################################

    def build_graph(self):
        self.define_placeholders()
        self.define_forward_pass()
        self.build_action_sampling()
        if self.training:
            with tf.variable_scope('train', reuse=tf.AUTO_REUSE):
                if self.nn_baseline:
                    self.build_baseline_forward_pass()
                self.define_train_op()

    ##################################

    def define_placeholders(self):
        raise NotImplementedError

    def define_forward_pass(self):
        if self.discrete:
            logits_na = build_mlp(self.observations_pl, output_size=self.ac_dim, scope='discrete_logits', n_layers=self.n_layers, size=self.size)
            self.parameters = logits_na
        else:
            mean = build_mlp(self.observations_pl, output_size=self.ac_dim, scope='continuous_logits', n_layers=self.n_layers, size=self.size)
            logstd = tf.Variable(tf.zeros(self.ac_dim), name='logstd')
            self.parameters = (mean, logstd)

    def build_action_sampling(self):
        if self.discrete:
            logits_na = self.parameters
            self.sample_ac = tf.squeeze(tf.multinomial(logits_na, num_samples=1), axis=1)
        else:
            mean, logstd = self.parameters
            self.sample_ac = mean + tf.exp(logstd) * tf.random_normal(tf.shape(mean), 0, 1)

    def define_train_op(self):
        raise NotImplementedError

    def define_log_prob(self):
        if self.discrete:
            #log probability under a categorical distribution
            logits_na = self.parameters
            self.logprob_n = tf.distributions.Categorical(logits=logits_na).log_prob(self.actions_pl)
        else:
            #log probability under a multivariate gaussian
            mean, logstd = self.parameters
            self.logprob_n = tfp.distributions.MultivariateNormalDiag(
                loc=mean, scale_diag=tf.exp(logstd)).log_prob(self.actions_pl)

    def build_baseline_forward_pass(self):
        self.baseline_prediction = tf.squeeze(build_mlp(self.observations_pl, output_size=1, scope='nn_baseline', n_layers=self.n_layers, size=self.size))

    ##################################

    def save(self, filepath):
        self.policy_saver.save(self.sess, filepath, write_meta_graph=False)

    def restore(self, filepath):
        self.policy_saver.restore(self.sess, filepath)

    ##################################

    # update/train this policy
    def update(self, observations, actions):
        raise NotImplementedError

    # query the neural net that's our 'policy' function, as defined by an mlp above
    # query the policy with observation(s) to get selected action(s)
    def get_action(self, obs):

        # TODO: GETTHIS from HW1

#####################################################
#####################################################

# class MLPPolicySL(MLPPolicy):

    # TODO: GETTHIS from HW1 (or comment it out, since you don't need it for this homework)

#####################################################
#####################################################

class MLPPolicyPG(MLPPolicy):
    # TODO: GETTHIS from HW2

#####################################################
#####################################################

class MLPPolicyAC(MLPPolicyPG):
    """ MLP policy required for actor-critic.

    Note: Your code for this class could in fact the same as MLPPolicyPG, except the neural net baseline
    would not be required (i.e. self.nn_baseline would always be false. It is separated here only
    to avoid any unintended errors. 
    """
    def __init__(self, *args, **kwargs):
        if 'nn_baseline' in kwargs.keys():
            assert kwargs['nn_baseline'] == False, "MLPPolicyAC should not use the nn_baseline flag"
        super().__init__(*args, **kwargs)
