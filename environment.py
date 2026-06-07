import random

class GridWorldEnv:
    def __init__(self, grid=None, gamma=0.99, is_slippery=True):
        if grid is None:
            grid = [
                "SFFF",
                "FHFH",
                "FFFH",
                "HFFG"
            ]
        self.grid = grid 
        self.n_rows = len(grid)
        self.n_cols = len(grid[0])

        self.states = list(range(self.n_rows * self.n_cols))
        self.actions = ["LEFT", "RIGHT", "UP", "DOWN"]
        self.start_state = self.find_start_state()
        self.state = self.start_state
        self.gamma = gamma
        self.is_slippery = is_slippery
        self.terminal_states = self.get_terminal_states()
    def state_to_pos(self, state):
        """
        Convert integer state into (row, col)
        """
        row = state // self.n_cols
        col = state % self.n_cols
        return row, col
    
    def pos_to_state(self, row, col):
        """
        Convert (row, col) back into integer state.
        
        """
        state = (row * self.n_cols) + col 
        return state

    def get_tile(self, state):
        """
        Return the character at a given state

        Example:
        "S", "F", "H", or "G"
        """
        row, col = self.state_to_pos(state)
        return self.grid[row][col]

    def find_start_state(self):
        """
        Find where the "S" tile i.
        
        """
        for s in self.states:
            tile = self.get_tile(s)
            if tile == "S":
                return s
            
        raise ValueError("No start state S found!")
    
    def get_terminal_states(self):
        """
        Returns a list of all terminal states.

        Terminal states are:
        - holes: "H"
        - goal: "G""

        """
        terminal_states = []
        for s in self.states:
            tile = self.get_tile(s)
            if tile in ["H", "G"]:
                terminal_states.append(s)
        return terminal_states
    def is_terminal(self, state):
        """
        Returns True if state is terminal, False otherwise.
        
        """
        return state in self.terminal_states
    def reward(self, next_state):
        """
        Return reward for entering next_state

        - Entering goal gives +1
        - Everything else gives 0
        
        """
        reward = 0
        next_tile = self.get_tile(next_state)
        if next_tile == "G":
            reward = 1
            return reward
        return reward
    def reset(self):
        """
        Reset agent back to start state.

        Return starting state

        """
        self.state = self.start_state
        return self.state
    def move(self, state, action):
        """
        Deterministically move from state to state according to action.
        Actions:
        - LEFT
        - RIGHT
        - UP 
        - DOWN

        Keep the agent inside the grid
        
        """
        row, col = self.state_to_pos(state)
        if action == "UP":
            row -= 1
        elif action == "DOWN":
            row += 1
        elif action == "RIGHT":
            col += 1
        elif action == "LEFT":
            col -= 1
        # I need to clamp row and col so they stay inside of the grid
        row = max(0, min(row, self.n_rows - 1))
        col = max(0, min(col, self.n_cols - 1))
        next_state = self.pos_to_state(row, col)
        
        return next_state

    def get_slippery_actions(self, action):
        """
        Returns the possible actual actions caused by slipping.
        """

        slippery_actions = {
        "LEFT": ["UP", "LEFT", "DOWN"],
        "RIGHT": ["UP", "RIGHT", "DOWN"],
        "UP": ["LEFT", "UP", "RIGHT"],
        "DOWN": ["LEFT", "DOWN", "RIGHT"]
        }

        return slippery_actions[action] # List of actions caused by slipping
    def get_transitions(self, state, action):
        """ 
        Return possible transitions from state after choosing action.

        Return:
        [(probability, next_state), ...]

        If the state is terminal:
            return [(1.0, state)]
        If not slippery:
            return one transition with probability 1.0.
        If slippery:
            Each actual action has probability 1/3.
        """
        if self.is_terminal(state) == True:
            return [(1.0, state)]
        if self.is_slippery == False:
            next_state = self.move(state, action)
            return [(1.0, next_state)]
        slippery_actions = self.get_slippery_actions(action)
        transitions = []
        for action in slippery_actions:
            transitions.append((1/3, self.move(state, action)))
        return transitions
    def step(self, action):
        """
        Take one sampled step in the environment.

        Return next_state, reward, done
        
        """
        # quit if terminal
        if self.is_terminal(self.state):
            return self.state, 0, True
        current_state = self.state
        transitions = self.get_transitions(current_state, action) # returns [(prob, state)...]
        # seperate probabilities and states
        probablities = [t[0] for t in transitions]
        states = [t[1] for t in transitions]
        next_state= random.choices(states, probablities)[0] # Sample from probabilities
        self.state = next_state
        # get reward
        r = self.reward(self.state)
        done = self.is_terminal(self.state)
        return self.state, r, done
    def render(self):
        """
        Print the grid.
        Shows agent as "A".
        
        Example:
        A F F F
        F H F H
        F F F H
        H F F G
        
        
        """
        for row in range(self.n_rows):
            row_string = ""
            for col in range(self.n_cols):
                state = self.pos_to_state(row, col)
                if state == self.state:
                    row_string += "A "
                else:
                    tile = self.get_tile(state)
                    row_string += tile + " "
            print(row_string)
