import numpy as np
import tensorflow as tf
import scipy
import scipy.stats

#Functions for further useage

class RunningStats(object):
    def __init__(self, N):
        self.N = N
        self.vals = []
        self.num_filled = 0

    def push(self, val):
        if self.num_filled == self.N:
            self.vals.pop(0)
            self.vals.append(val)
        else:
            self.vals.append(val)
            self.num_filled += 1

    def push_list(self, vals):
        num_vals = len(vals)

        self.vals.extend(vals)
        self.num_filled += num_vals
        if self.num_filled >= self.N:
            diff = self.num_filled - self.N
            self.num_filled = self.N
            self.vals = self.vals[diff:]

    def get_mean(self):
        return np.mean(self.vals[:self.num_filled])

    def get_std(self):
        return np.std(self.vals[:self.num_filled])

    def get_mean_n(self, n):
        start = max(0, self.num_filled-n)
        return np.mean(self.vals[start:self.num_filled])
    

def multinomial_rvs(n, p):
    """
    Sample from the multinomial distribution with multiple p vectors.

    * n must be a scalar.
    * p must an n-dimensional numpy array, n >= 1.  The last axis of p
      holds the sequence of probabilities for a multinomial distribution.

    The return value has the same shape as p.
    """
    count = np.full(p.shape[:-1], n)
    out = np.zeros(p.shape, dtype=int)
    ps = p.cumsum(axis=-1)
    # Conditional probabilities
    with np.errstate(divide='ignore', invalid='ignore'):
        condp = p / ps
    condp[np.isnan(condp)] = 0.0
    for i in range(p.shape[-1]-1, 0, -1):
        binsample = np.random.binomial(count, condp[..., i])
        out[..., i] = binsample
        count -= binsample
    out[..., 0] = count
    return out


def multinomial_rvs1(n, p):
    """
    Sample from the multinomial distribution with multiple p vectors.

    * n must be a scalar.
    * p must an n-dimensional numpy array, n >= 1.  The last axis of p
      holds the sequence of probabilities for a multinomial distribution.

    The return value has the same shape as p.
    """
    count = np.full(p.shape[:-1], n)
    out = np.zeros(p.shape, dtype=int)
    ps = p.cumsum(axis=-1)
    # Conditional probabilities
    with np.errstate(divide='ignore', invalid='ignore'):
        condp = p / ps
    condp[np.isnan(condp)] = 0.0
    while count > 0:
        for i in range(p.shape[-1]-1, 0, -1):
            binsample = np.random.binomial(1, condp[..., i])
            out[..., i] = binsample
            count -= binsample
    out[..., 0] = count
    return out

# Different policy distributions

class ProbabilityAction(object):
    def __init__(self, num_param, action_dim):
        """
        Class that implements various probabilistic actions

        Args:
            num_param (TYPE): number of parameters for a single distribution (P)
            example: single variable gaussian will have two parameters: mean and variance
        """
        self.num_param = num_param
        self.action_dim = action_dim

    def log_prob(self, params, selected_action):
        """
        Given a batch of distribution parameters
        and selected actions.
        Compute log probability of those actions

        Args:
            params (TYPE): N by (A*P) Tensor
            selected_action (TYPE): N by A Tensor

        Returns:
            Length N Tensor of log probabilities, N by (A*P) Tensor of corrected parameters
        """
        raise Exception("Not Implemented")

    def get_action(self, params):
        """

        Args:
            params (TYPE): N by (A*P) Numpy array

        Returns: N by A Numpy array of sampled actions
        """
        raise Exception("Not Implemented")

    def get_mean_action(self, params):
        """

        Args:
            params (TYPE): N by (A*P) Numpy array

        Returns: N by A Numpy array of sampled actions
        """
        raise Exception("Not Implemented")

    def random_action(self, N):
        """

        Args:
            params (TYPE): N by (A*P) Numpy array

        Returns: N by A Numpy array of sampled actions
        """
        raise Exception("Not Implemented")
        
# Truncated Gaussian distribution for power allocation       

class TruncatedGaussianDistribution(ProbabilityAction):
    def __init__(self, action_dim, lower_bound, upper_bound):
        super().__init__(2, action_dim)

        if lower_bound.shape != (action_dim,) or \
           upper_bound.shape != (action_dim,):
           raise Exception("Lower and upperbounds not the right shape")
        self.lower_bound = np.array(lower_bound, dtype=np.float32)
        self.upper_bound = np.array(upper_bound, dtype=np.float32)


    def log_prob(self, params, selected_action):
        mean = tf.gather(params, np.array(range(self.action_dim)), axis=1, name='mean')
        std = tf.gather(params, np.array(range(self.action_dim, 2*self.action_dim)), axis=1, name='std')

        mean = tf.nn.sigmoid(mean) * (self.upper_bound - self.lower_bound) + self.lower_bound
        std = tf.nn.sigmoid(std) * np.sqrt(self.upper_bound - self.lower_bound) + 0.05 # TODO: add a little epsilon?

        output = tf.concat([mean, std], axis=1)

        dist = tf.distributions.Normal(mean, std)

        log_probs = dist.log_prob(selected_action) - tf.log(dist.cdf(self.upper_bound) - dist.cdf(self.lower_bound))
        log_probs = tf.reduce_sum(log_probs, axis=1)
        return log_probs, output

    def get_action(self, params):
        mean = params[:, :self.action_dim]
        std = params[:, self.action_dim:]

        N = params.shape[0]

        lower_bound = (np.vstack([self.lower_bound for _ in range(N)]) - mean) / std
        upper_bound = (np.vstack([self.upper_bound for _ in range(N)]) - mean) / std

        action = scipy.stats.truncnorm.rvs(lower_bound, upper_bound, loc=mean, scale=std)

        return action

    def get_mean_action(self, params):
        mean = params[:, :self.action_dim]
        std = params[:, self.action_dim:]

        action = mean

        return action

    def random_action(self, N):
        action = np.random.uniform(low=self.lower_bound, high=self.upper_bound,size=(N,self.action_dim))

        return action

# Category distribution for relay selection

class CategoricalDistribution(ProbabilityAction):
    def __init__(self, action_dim, num_classes):
        super().__init__(1, action_dim)
        self.num_classes = num_classes

    def log_prob(self, params, selected_action):
        N = tf.shape(params)[0]

        params2 = tf.reshape(params,(N,-1,self.num_classes))
        ru_class = tf.nn.softmax(params2,axis=2)

        output = tf.reshape(ru_class,(N,self.action_dim))

        output2 = tf.math.maximum(output,1e-5)

        log_probs = selected_action * tf.math.log(output2)
        log_probs = tf.reduce_sum(log_probs,axis=1)

        return log_probs, output


    def get_action(self, params):
        output_list = []
        N = np.shape(params)[0]

        params2 = np.reshape(params,(N,-1,self.num_classes))
        action = multinomial_rvs(1,params2)

        return np.reshape(action,(N,self.action_dim))

    def random_action(self,N):
        output_list = []
        probs = np.ones((self.num_classes))/self.num_classes
        action = np.random.multinomial(1,probs,(N,int(self.action_dim/self.num_classes)))

        return np.reshape(action,(-1,self.action_dim))

    def get_mean_action(self, params):
        output_list = []
        N = np.shape(params)[0]

        params2 = np.reshape(params,(N,-1,self.num_classes))
        action = np.zeros_like(params2)
        action[np.arange(len(params2)), params2.argmax(1)] = 1

        return np.reshape(action,(N,self.action_dim))

#Combination for joint power and relay allocation

class CategoryGaussianDistribution(ProbabilityAction):
    def __init__(self, action_dim, num_relays, num_channels, lower_bound, upper_bound):
        super(CategoryGaussianDistribution,self).__init__(12/7, action_dim)
        
        self.num_relays = num_relays
        self.num_channels = num_channels              
        
        self.gaussian1 = TruncatedGaussianDistribution(num_channels, lower_bound, upper_bound)
        self.gaussian2 = TruncatedGaussianDistribution(num_channels, lower_bound, upper_bound)        
        
        self.catago1 = CategoricalDistribution(num_relays, num_relays)
        

    def log_prob(self, params, selected_action):

        pt1 = int(self.num_channels)
        
        pt2 = int(self.num_relays)
        
        gaussian1_p = tf.gather(params, np.array(range(2*pt1)), axis=1, name='gp1')
        gaussian2_p = tf.gather(params, np.array(range((2*pt1), (4*pt1))), axis=1, name='gp2')
        
        
        catago1_p = tf.gather(params, np.array(range(4*pt1, (4*pt1 + pt2))), axis=1, name='bp1')       
        
        transmit_action1 = tf.gather(selected_action, np.array(range(pt1)), axis=1, name='transmit1')
        transmit_action2 = tf.gather(selected_action, np.array(range((pt1),(2*pt1))), axis=1, name='transmit2')        
        
        relay_action1 = tf.gather(selected_action, np.array(range(pt1, (pt1 + pt2))), axis=1, name='relay1')

        log_probs1, output1 = self.gaussian1.log_prob(gaussian1_p,transmit_action1)
        log_probs2, output2 = self.gaussian2.log_prob(gaussian2_p,transmit_action2)
        
        log_probs11, output11 = self.catago1.log_prob(catago1_p,relay_action1)

        output = tf.concat([output1, output2, output11], axis=1)

        log_probs = log_probs1 + log_probs2
        log_probs = log_probs + log_probs11
        

        return log_probs, output

    def get_action(self, params):
        
        pt1 = int(self.num_channels)
        pt2 = self.num_relays

        gaussian1_p = params[:,:2*pt1]
        gaussian2_p = params[:,(2*pt1):(4*pt1)]
        catago1_p = params[:,(4*pt1):(4*pt1 + pt2)]

        action1 = self.gaussian1.get_action(gaussian1_p)
        action2 = self.gaussian2.get_action(gaussian2_p)
        
        action11 = self.catago1.get_action(catago1_p)
        
        action = np.concatenate([action1, action2, action11], axis=1)

        return action

    def get_mean_action(self, params):
        pt1 = int(self.action_dim)
        pt2 = int(3/2*self.action_dim)

        gaussian_p = params[:,:pt1]
        bernoulli_p = params[:,pt1:pt2]

        action1 = self.gaussian.get_mean_action(gaussian_p)
        action2 = self.bernoulli.get_mean_action(bernoulli_p)

        action = np.concatenate([action1, action2], axis=1)

        return action

    def random_action(self, N):

        action1 = self.beta.random_action(N)
        action2 = self.bernoulli.random_action(N)

        action = np.concatenate([action1, action2], axis=1)

        return action

# Define DNN model

def mlp_model(state_dim, action_dim, num_param):
    with tf.variable_scope('policy'):
        state_input = tf.placeholder(tf.float32, [None, state_dim], name='state_input')

        layer1 = tf.contrib.layers.fully_connected(state_input,
            200,
            activation_fn=tf.nn.relu,
            scope='layer1')

        layer2 = tf.contrib.layers.fully_connected(layer1,
            100,
            activation_fn=tf.nn.relu,
            scope='layer2')

        output = tf.contrib.layers.fully_connected(layer2,
            int(60),
            activation_fn=None,
            scope='output')

    return state_input, output

# Primal-dual learning function

class LearningPolicy(object):
    def __init__(self,
        state_dim,
        action_dim,
        constraint_dim,
        model_builder=mlp_model,
        distribution=None,
        sess=None):

        self.state_dim = state_dim
        self.action_dim = action_dim
        self.constraint_dim = constraint_dim
        self.lambd = np.ones((constraint_dim, 1))*1

        self.model_builder = mlp_model
        self.dist = distribution

        self.lambd_lr = 0.0025
        self.stats = RunningStats(64*100)
        self._build_model(state_dim, action_dim, self.model_builder, distribution)

        if sess == None:
            config = tf.ConfigProto(device_count={'GPU': 0})
            self.sess = tf.InteractiveSession(config=config)
            tf.global_variables_initializer().run()
        else:
            self.sess = sess


    def _build_model(self, state_dim, action_dim, model_builder, distribution):

        self.state_input, self.output = model_builder(state_dim, action_dim, self.dist.num_param)

        self.selected_action = tf.placeholder(tf.float32, [None, action_dim], name='selected_action')

        self.log_probs, self.params = self.dist.log_prob(self.output, self.selected_action)

        self.cost = tf.placeholder(tf.float32, [None], name='cost')

        self.loss = self.log_probs * self.cost
        
        self.loss = tf.reduce_mean(self.loss)

        lr = 5e-5
        self.optimize = tf.train.AdamOptimizer(lr).minimize(self.loss)


    def get_action(self, inputs):
        
        fd = {self.state_input: inputs}

        params = self.sess.run(self.params, feed_dict=fd)

        action = self.dist.get_action(params)
        
        return action

    def learn(self, inputs, actions, f0, f1):
        """
        Args:
            inputs (TYPE): N by m
            actions (TYPE): N by m
            f0 (TYPE): N by 1
            f1 (TYPE): N by p

        Returns:
            TYPE: Description
        """

        cost = f0 + np.dot(f1, self.lambd)
        cost = np.reshape(cost, (-1))

        self.stats.push_list(cost)
        cost_minus_baseline = cost - self.stats.get_mean()

        # policy gradient step
        fd = {self.state_input: inputs,
              self.selected_action: actions,
              self.cost: cost_minus_baseline}

        output = self.sess.run(self.output, feed_dict=fd)

        loss, _ = self.sess.run([self.loss, self.optimize], feed_dict=fd)

        # gradient ascent step on lambda
        delta_lambd = np.mean(f1, axis=0)
        
        delta_lambd = np.reshape(delta_lambd, (-1, 1))

        self.lambd += delta_lambd * self.lambd_lr
        
        # project lambd into positive orthant
        self.lambd = np.maximum(self.lambd, 0)

        self.lambd_lr *= 0.99995

        return loss
