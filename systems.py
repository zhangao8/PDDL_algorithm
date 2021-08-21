import numpy as np

# Define FSO systems

class ProbabilisticSystem(object):
    def __init__(self, state_dim, action_dim, constraint_dim):

        self.state_dim = state_dim
        self.action_dim = action_dim
        self.constraint_dim = constraint_dim

    def _lognormal_sample(self,batch_size):
        samples = np.random.lognormal(self.mu, self.sigma, size=(batch_size, self.state_dim))
        return samples
    
class FSOCapacitySystem(ProbabilisticSystem):
    def __init__(self, num_channels, hl, num_relays, PS, PT, e, A, r, Tf, B, w=None, v=None, mu=0, sigma=0.2):
        super(FSOCapacitySystem,self).__init__(2*num_relays*num_channels, 3*num_channels, 2)
        
        # Define system parameters
        self.num_relays = num_relays
        self.num_channels = num_channels
        self.PT = PT
        self.PS = PS
        self.mu = mu
        self.sigma = sigma
        self.r = r
        self.hl = hl
        self.B=B
        self.Tf = Tf
        self.e = e
        self.A = A
        self.v = np.ones((1, num_channels))*0.05
        self.w = np.random.random_sample((1, num_channels))                        
        self.w[0,0]=0.56        
        self.w[0,1]=0.28        
        self.w[0,2]=0.8        
        self.w[0,3]=0.01        
        self.w[0,4]=0.03
        
    def sample(self, batch_size):
        
        # Generate channel state information
        return self._lognormal_sample(batch_size)

    def f0(self, state, action):

        # Compute the objective (e.g. capacity formula, etc.)
        D = np.hsplit(state, self.num_relays)
        capacity = np.zeros((np.size(D[0],0),1))

        for i in np.arange(np.size(D[0],0)):
                
            A1 = action[:,:self.num_channels]
            A2 = action[:,self.num_channels:(2*self.num_channels)]
            A3 = action[:,(2*self.num_channels):(2*self.num_channels+self.num_relays)]
                
            indictor=-1
            j=0
            while indictor==-1:
                if A3[i,j] == 1:
                    indictor = j
                j = j+1
            
            state_m = D[indictor]
            state1 = state_m[:,:self.num_channels]
            state2 = state_m[:,self.num_channels:(2*self.num_channels)]
            state_m1 = state1.reshape((np.size(state_m,0),self.num_channels))
            state_m2 = state2.reshape((np.size(state_m,0),self.num_channels))

            capacity_temp = self.w*self.Tf*self.B*np.log(1+1 / ((1+1/A1[i,:]/state_m1[i,:]/state_m1[i,:]/self.hl/self.hl/self.r/self.A*self.e*self.B)*(1+1/A2[i,:]/state_m2[i,:]/state_m2[i,:]/self.hl/self.hl/self.r/self.A*self.e*self.B)-1))
            capacity[i,0] = capacity[i,0]-np.sum(capacity_temp)          
            
        return capacity
    

    def f1(self, state, action):

        # Compute the constraints
        A1 = action[:,:self.num_channels]
        A2 = action[:,self.num_channels:(2*self.num_channels)]
        
        cons1 = np.array([np.sum(A1, axis=1) - self.PT])
        cons2 = np.array([np.sum(A2, axis=1) - self.PT])
        cons1 = cons1.T
        cons2 = cons2.T
        cons = np.concatenate((cons1,cons2),axis=1)
        
        return cons


