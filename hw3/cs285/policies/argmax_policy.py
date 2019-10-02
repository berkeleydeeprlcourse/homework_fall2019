import tensorflow as tf

class ArgMaxPolicy(object):

    def __init__(self, sess, critic):
        self.sess = sess
        self.critic = critic

        # TODO: Define what action this policy should return
        # HINT1: the critic's q_t_values indicate the goodness of observations, 
        # so they should be used to decide the action to perform
        self.action = tf.argmax(TODO, axis=1)

    def get_action(self, obs):

        # TODO: Make use of self.action by passing these input observations into self.critic
        # HINT: you'll want to populate the critic's obs_t_ph placeholder 
        if len(obs.shape) > 1:
            observation = obs
        else:
            observation = obs[None]
        return TODO