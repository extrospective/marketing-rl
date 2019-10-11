import numpy as np
import random

### Interface
class Environment(object):

    def reset(self):
        raise NotImplementedError('Inheriting classes must override reset.')

    def actions(self):
        raise NotImplementedError('Inheriting classes must override actions.')

    def step(self):
        raise NotImplementedError('Inheriting classes must override step')

        
class ActionSpace(object):
    
    def __init__(self, actions):
        self.actions = actions
        self.n = len(actions)
        

class SimpleCustomerPurchase():
    
    def __init__(self, tdeath=11, tmin=5, tmax=10, prob_response=0.2, gross_profit_response=100):
        self.tmin = tmin
        self.tmax = tmax
        self.tdeath = tdeath
        self.prob_response = prob_response
        self.gross_profit_response = gross_profit_response
        
    def get_reward(self, t_state):
        if (t_state >= self.tmin) and (t_state < self.tmax):
            if random.random() < self.prob_response:
                return self.get_gross_profit()
        return 0
    
    def get_gross_profit(self):
        return self.gross_profit_response
    
    def get_death(self, t_state):
        if t_state >= self.tdeath:
            return True
        return False
    
    def get_next_state(self, t_state):
        return t_state + 1
    
    def get_max_states(self):
        return self.tdeath
    
    def get_eligibility_window(self):
        return (self.tmin, self.tmax)
    
    
class CustomerPurchaseVariable(SimpleCustomerPurchase):
    
    def __init__(self, tdeath, tmin, tmax, prob_response, gross_profit_response_max):
        super(CustomerPurchaseVariable, self).__init__(tdeath, tmin, tmax, prob_response)
        self.gross_profit_response_max = gross_profit_response_max
        
    def get_gross_profit(self):
        spend = random.randint(0, self.gross_profit_response_max)
        # print(spend)
        return spend
    
        
class SimpleMarketingEnv(Environment):

    def __init__(self, customer_class, cost_marketing_action=1, max_steps = 5000):
        super(SimpleMarketingEnv, self).__init__()

        # define state and action space
        self.cust = customer_class

        self.S = range(self.cust.get_max_states())
        self.action_space = ActionSpace(range(2))

        self.max_steps = max_steps
        self.cost_marketing_action = cost_marketing_action
        self.elig_min, self.elig_max = self.cust.get_eligibility_window()

        
    def step(self, action):
        s_prev = self.s
        purchase = 0
        self.nstep += 1
        if self.nstep > self.max_steps:
            print('Aborting due to max steps %d reached' % self.max_steps)
            raise('too many steps')
        self.s = self.cust.get_next_state(s_prev)
        if action == 0:
            # no marketing action
            reward = 0
        else:
            # Marketing action
            reward = - self.cost_marketing_action
            mkt_reward = self.cust.get_reward(self.s)
            if mkt_reward > 0:
                reward += mkt_reward
                self.s = 0

        # Are we at the end for the customer lifetime?
        if self.cust.get_death(self.s):
            is_reset = True
            self.reset()
        else:
            is_reset = False
                
        return (self._convert_state(self.s), reward, is_reset, '')
    
    
    def reset(self):
        self.nstep = 0
        self.s = 0
        self.is_reset = True
        return self._convert_state(self.s)
    
    
    def _convert_state(self, s):
        converted = np.zeros(len(self.S), dtype=np.float32)
        converted[s] = 1
        return converted
    

    def render(self, mode='rgb_array', close=False):
        if close:
            return

        if mode == 'rgb_array':
            # 2D: one high, and width = self.tdeath
            
            # default 1's: yellow
            # default 0's: purple
            maze = np.ones((1, self.cust.get_max_states()))
            #
            # color area where one can market
            #
            maze[0, self.elig_min:self.elig_max+1] = .8
            
            #
            # color the agent: where are they located
            maze[(0, self.s)] = 0
            
            return np.array(maze, copy=True)
        else:
            print('Not rendering non-rgb_array')


