import numpy as np
import random
from itertools import cycle

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
        
    
        
class SimpleMarketingEnv(Environment):

    def __init__(self, customer_list, cost_marketing_action=1, max_steps = 5000):
        super(SimpleMarketingEnv, self).__init__()

        # define state and action space
        self.customer_list = customer_list
        self.customer_list_cycle = cycle(customer_list)
        
        max_states = 0
        for i in self.customer_list:
            max_states = i.get_max_states() if i.get_max_states() > max_states else max_states
        self.shape = (1, max_states)
        self.S = range(max_states)
        self.action_space = ActionSpace(range(2))

        print('In this environment, actions 0=no marketing action, 1=marketing action')
        print('Marketing actions will only work in a range of possible steps.')
        
        self.max_steps = max_steps
        self.cost_marketing_action = cost_marketing_action

        
    def step(self, action):
        s_prev = self.s
        purchase = 0
        self.nstep += 1
        if self.nstep > self.max_steps:
            print('Aborting due to max steps %d reached' % self.max_steps)
            raise('too many steps')
            
        # moves to next state.  If the customer suddenly died that happens here
        # this assumes that the system knows the customer has died! which is not true in reality.
        self.s = self.cust.get_next_state(s_prev)
        # no marketing action
        reward = 0
        if action >= 1:
            # Marketing action
            reward = - self.cost_marketing_action
            
            # additional reward may occur; and reset to state 0
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
        #
        # cycle through customers one at a time
        self.cust = next(self.customer_list_cycle)
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
            maze = np.ones((2, self.shape[1]))
            #
            # color area where one can market
            #
            self.elig_min, self.elig_max = self.cust.get_eligibility_window()
            maze[0, self.elig_min:self.elig_max] = .8
            
            #
            # color the agent: where are they located
            maze[(0, self.s)] = 0
            
            return np.array(maze, copy=True)
        else:
            print('Not rendering non-rgb_array')


            
class TwoValueMarketingEnv(Environment):

    def __init__(self, customer_list, cost_marketing_action=1, p_initial_high=0.3, 
                 p_low_to_high=0.5, p_high_to_low=0.1, max_steps=5000):
        super(TwoValueMarketingEnv, self).__init__()

        # define state and action space
        self.customer_list = customer_list
        self.customer_list_cycle = cycle(customer_list)
        self.cost_marketing_action = cost_marketing_action
        self.p_initial_high = p_initial_high
        self.p_low_to_high = p_low_to_high
        self.p_high_to_low = p_high_to_low
                
        #
        # Normally one would ask the agent to contain the "guess" about the customer
        # In this experiment we are asking what would happen if the environment presented
        # a more complex state: high and low value customers as it made a guess based on
        # the reward pattern
        #
        # p_initial_high = probability that an episode is initialized in high state
        # p_low_to_high = probability that a purchaser in a low state will be deemed high value state
        # p_high_to_low = probability that a non-purchaser in a high value state will be deemed low value state
        
        #
        # number of states: There are two sets of states avaiable: 
        # 
        # HighValueStates and LowValueStates
        # 
        # these are based on an estimate from the environment about what state we are in
        #
        # HighValueState = state where customer is believed to be a high value customer
        # LowValueState = state where customer is believe to be a low value customer  
        
        # 
        # state is maximum of any customer.  although we do not ensure what happens in the customer class with different
        # numbers of states are passed in.
        max_states = 0
        for i in self.customer_list:
            max_states = i.get_max_states() if i.get_max_states() > max_states else max_states
        self.shape = (2, max_states)
        self.S = np.zeros(self.shape, dtype=np.bool)
        #
        # still two actions: market or do not market at time T
        self.action_space = ActionSpace(range(2))

        print('In this two value marketing environment, actions 0=no marketing action, 1=marketing action')
        print('Marketing actions will only work in a range of possible steps.')
        print('Additionally there are high value and low value states provided by the environment')
        self.max_steps = max_steps
        self.reset()
        
    def _convert_state(self, state):
        converted = np.unravel_index(state, self.shape)
        return np.asarray(list(converted), dtype=np.float32)

    def step(self, action):
        s_prev = self.s
        purchase = 0
        #print('taking step')
        self.nstep += 1
        if self.nstep > self.max_steps:
            print('Aborting due to max steps %d reached' % self.max_steps)
            raise('too many steps')
            
        # moves to next state.  If the customer suddenly died that happens here
        #
        #  One flaw is we do not see the cost of marketing to the dead customer I think
        #
        self.s = self.cust.get_next_state(s_prev)
        reward = 0
        if action >= 1:
            # Marketing action
            #print('marketing action occurred')
            reward = - self.cost_marketing_action
            
            # additional reward may occur; and reset to state 0
            mkt_reward = self.cust.get_reward(self.s)
            if mkt_reward > 0:
                # purchased
                reward += mkt_reward
                self.s = 0
                if self.is_low == 1:
                    if random.random() < self.p_low_to_high:
                        #print('moved to high')
                        self.is_low = 0
            else:
                # not purchased.  may transition from high to low
                if self.is_low == 0:
                    if random.random() < self.p_high_to_low:
                        #print('moved to low')
                        self.is_low = 1

        # Are we at the end for the customer lifetime?
        if self.cust.get_death(self.s):
            is_reset = True
            self.reset()
        else:
            is_reset = False
        #print('step done')  
        return (self._convert_state(self.s, self.is_low), reward, is_reset, '')
 
    
    def reset(self):
        #
        # Each pass through a different customer is selected
        # But the agent is not given any indication from state which type of customer it is
        # This simple model tries to simply model states based on customer behavior rather than leak any
        # atual customer state into the environment.
        self.nstep = 0
        self.s = 0
        self.is_reset = True
        
        #
        # cycle through customers one at a time
        self.cust = next(self.customer_list_cycle)
        
        #
        # This is the independent setting of state in the environment
        # randomize high/low
        self.is_low = 0 if random.random() < self.p_initial_high else 1
        return self._convert_state(self.s, self.is_low)
    
    
    def _convert_state(self, state, is_low):
        return np.asarray((is_low, state), dtype=np.float32)
    

    def render(self, mode='rgb_array', close=False):
        if close:
            return

        if mode == 'rgb_array':
            # 2D: two high, and width = self.tdeath
            # higher row should be high value
            
            # default 1's: yellow
            # default 0's: purple
            maze = np.ones((2, self.shape[1]))
            #
            # color area where one can market: note could cause flickering if different for customers
            # maybe in future the likelihood of response factors into the color too.
            #
            self.elig_min, self.elig_max = self.cust.get_eligibility_window()
            maze[:, self.elig_min:self.elig_max] = .8
            
            #
            # color the agent: where are they located
            maze[(self.is_low, self.s)] = 0
            #print('rendering array')
            #print(maze)
            return np.array(maze, copy=True)
        else:
            print('Not rendering non-rgb_array')

            

            
class Customer():
    
    def __init__(self, gross_profit_min, gross_profit_max):
        self.gross_profit_min = gross_profit_min
        self.gross_profit_max = gross_profit_max
        
    def get_gross_profit(self):
        spend = random.randint(self.gross_profit_min, self.gross_profit_max)
        return spend

           
        
class CustomerInventoryDriven(Customer):
    """
    CustomerInventoryDriven -- a customer who responds in time window after purchase only
    """
    def __init__(self, gross_profit_min=50, gross_profit_max=200, tdeath=11, tmin=5, tmax=10, prob_response=0.2,  
                 own_purchase_prob=0,
                 prob_sudden_death=0.001):
        super().__init__(gross_profit_min, gross_profit_max)
        """
        tmin: min time for open to marketing action
        tmax: one past time open to marketing action
        prob_response: probability of response to marketing action
        gross_profit_response: GP generated by purchase
        own_purchase_prob: probability of making a purchase on their own not driven by marketing
            (note: this will appear to be a reward to the system since driver for it is not known to agent)
            (customer which responds with purchase does not also make its own purchase)
        prob_sudden_death: probability that customer dies in current state and all built up value is gone
        """
        self.tdeath = tdeath
        self.tmin = tmin
        self.tmax = tmax
        self.prob_response = prob_response
        self.own_purchase_prob = own_purchase_prob
        self.prob_sudden_death = prob_sudden_death
        self.dead = False
        
    def get_reward(self, t_state):
        if self.dead:
            # no reward for the dead
            return 0

        # random unrelated purchases
        if random.random() < self.own_purchase_prob:
            return self.get_gross_profit()
        
        # purchases in response to marketing
        if (t_state >= self.tmin) and (t_state < self.tmax):
            if random.random() < self.prob_response:
                return self.get_gross_profit()

        # default: no purchase
        return 0
    
    
    def get_death(self, t_state):
        #print('checking get_death')
        if t_state >= self.tdeath:
            #print('state >= tdeath, tstate %d tdeath %d' % (t_state, self.tdeath))
            self.dead = False  # resurrect for next customer
            return True
        return False
    
    
    def get_next_state(self, t_state):
        if random.random() < self.prob_sudden_death:
            # customer suddenly dies but marketing cannot tell that
            # they just see no response
            self.dead = True
        return t_state + 1
    
    
    def get_max_states(self):
        return self.tdeath
    
    
    def get_eligibility_window(self):
        return (self.tmin, self.tmax)
    

        
