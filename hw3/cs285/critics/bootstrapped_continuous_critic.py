from .base_critic import BaseCritic
import tensorflow as tf
from cs285.infrastructure.tf_utils import build_mlp

class BootstrappedContinuousCritic(BaseCritic):
    def __init__(self, sess, hparams):
        self.sess = sess
        self.ob_dim = hparams['ob_dim']
        self.ac_dim = hparams['ac_dim']
        self.discrete = hparams['discrete']
        self.size = hparams['size']
        self.n_layers = hparams['n_layers']
        self.learning_rate = hparams['learning_rate']
        self.num_target_updates = hparams['num_target_updates']
        self.num_grad_steps_per_target_update = hparams['num_grad_steps_per_target_update']
        self.gamma = hparams['gamma']

        self._build()

    def _build(self):
        """
            Notes on notation:

            Symbolic variables have the prefix sy_, to distinguish them from the numerical values
            that are computed later in the function

            Prefixes and suffixes:
            ob - observation 
            ac - action
            _no - this tensor should have shape (batch self.size /n/, observation dim)
            _na - this tensor should have shape (batch self.size /n/, action dim)
            _n  - this tensor should have shape (batch self.size /n/)

            Note: batch self.size /n/ is defined at runtime, and until then, the shape for that axis
            is None

            ----------------------------------------------------------------------------------
            loss: a function of self.sy_ob_no, self.sy_ac_na and self.sy_adv_n that we will differentiate
                to get the policy gradient.
        """
        self.sy_ob_no, self.sy_ac_na, self.sy_adv_n = self.define_placeholders()

        # define the critic
        self.critic_prediction = tf.squeeze(build_mlp(
            self.sy_ob_no,
            1,
            "nn_critic",
            n_layers=self.n_layers,
            size=self.size))
        self.sy_target_n = tf.placeholder(shape=[None], name="critic_target", dtype=tf.float32)

        # TODO: set up the critic loss
        # HINT1: the critic_prediction should regress onto the targets placeholder (sy_target_n)
        # HINT2: use tf.losses.mean_squared_error
        self.critic_loss = TODO

        # TODO: use the AdamOptimizer to optimize the loss defined above
        self.critic_update_op = TODO

    def define_placeholders(self):
        """
            Placeholders for batch batch observations / actions / advantages in actor critic
            loss function.
            See Agent.build_computation_graph for notation

            returns:
                sy_ob_no: placeholder for observations
                sy_ac_na: placeholder for actions
                sy_adv_n: placeholder for advantages
        """
        sy_ob_no = tf.placeholder(shape=[None, self.ob_dim], name="ob", dtype=tf.float32)
        if self.discrete:
            sy_ac_na = tf.placeholder(shape=[None], name="ac", dtype=tf.int32)
        else:
            sy_ac_na = tf.placeholder(shape=[None, self.ac_dim], name="ac", dtype=tf.float32)
        sy_adv_n = tf.placeholder(shape=[None], name="adv", dtype=tf.float32)
        return sy_ob_no, sy_ac_na, sy_adv_n

    def forward(self, ob):
        # TODO: run your critic
        # HINT: there's a neural network structure defined above with mlp layers, which serves as your 'critic'
        return TODO

    def update(self, ob_no, next_ob_no, re_n, terminal_n):
        """
            Update the parameters of the critic.

            let sum_of_path_lengths be the sum of the lengths of the sampled paths
            let num_paths be the number of sampled paths

            arguments:
                ob_no: shape: (sum_of_path_lengths, ob_dim)
                next_ob_no: shape: (sum_of_path_lengths, ob_dim). The observation after taking one step forward
                re_n: length: sum_of_path_lengths. Each element in re_n is a scalar containing
                    the reward for each timestep
                terminal_n: length: sum_of_path_lengths. Each element in terminal_n is either 1 if the episode ended
                    at that timestep of 0 if the episode did not end

            returns:
                loss
        """

        # TODO: Implement the pseudocode below: 

        # do the following (self.num_grad_steps_per_target_update * self.num_target_updates) times:
            # every self.num_grad_steps_per_target_update steps (which includes the first step),
                # recompute the target values by 
                    #a) calculating V(s') by querying this critic network (ie calling 'forward') with next_ob_no
                    #b) and computing the target values as r(s, a) + gamma * V(s')
                # HINT: don't forget to use terminal_n to cut off the V(s') (ie set it to 0) when a terminal state is reached
            # every time,
                # update this critic using the observations and targets
                # HINT1: need to sess.run the following: 
                    #a) critic_update_op 
                    #b) critic_loss
                # HINT2: need to populate the following (in the feed_dict): 
                    #a) sy_ob_no with ob_no
                    #b) sy_target_n with target values calculated above
        
        TODO

        return loss
