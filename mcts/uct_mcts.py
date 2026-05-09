from .node import MCTSNode, INF
from .config import MCTSConfig
from env.base_env import BaseGame

import numpy as np

class UCTMCTSConfig(MCTSConfig):
    def __init__(
        self,
        n_rollout:int = 1,
        *args, **kwargs
    ):
        MCTSConfig.__init__(self, *args, **kwargs)
        self.n_rollout = n_rollout


class UCTMCTS:
    def __init__(self, init_env:BaseGame, config: UCTMCTSConfig, root:MCTSNode=None):
        self.config = config
        self.root = root
        if root is None:
            self.init_tree(init_env)
        self.root.cut_parent()
    
    def init_tree(self, init_env:BaseGame):
        # initialize the tree with the current state
        # fork the environment to avoid side effects
        env = init_env.fork()
        self.root = MCTSNode(
            action=None, env=env, reward=0,
        )
    
    def get_subtree(self, action:int):
        # return a subtree with root as the child of the current root
        # the subtree represents the state after taking action
        if self.root.has_child(action):
            new_root = self.root.get_child(action)
            return UCTMCTS(new_root.env, self.config, new_root)
        else:
            return None
    
    def uct_action_select(self, node:MCTSNode) -> int:
        # select the best action based on UCB when expanding the tree
        
        ########################
        # TODO: your code here #
        ########################
        valid_mask = node.action_mask == 1 # 转换为bool数组
        N_parent = np.sum(node.child_N_visit)

        q =  node.child_V_total / (node.child_N_visit + 1e-8)
        explore =  self.config.C * np.sqrt(np.log(N_parent) / (node.child_N_visit + 1e-8))
        ucb = q + explore

        ucb[~valid_mask] = -INF # 非法动作设为负无穷，确保不被选中

        return np.argmax(ucb)
        ########################

    def backup(self, node:MCTSNode, value:float) -> None:
        # backup the value of the leaf node to the root
        # update N_visit and V_total of each node in the path
        
        ########################
        # TODO: your code here #
        ########################
        while node.parent is not None:
            parent = node.parent
            action = node.action
            value = -value
            parent.child_N_visit[action] += 1
            parent.child_V_total[action] += value
            node = parent 
        ########################    
            
    
    def rollout(self, node:MCTSNode) -> float:
        # simulate the game until the end
        # return the reward of the game
        # NOTE: the reward should be convert to the perspective of the current player!
        
        ########################
        # TODO: your code here #
        ########################
        if node.done:
            return -node.reward # node.reward实际上是父节点的视角，所以需要取反
        
        env = node.env.fork()
        done = env.ended
        multiplier = 1
        while not done:
            valid_actions = np.where(env.action_mask == 1)[0]
            action = np.random.choice(valid_actions)
            _, reward, done = env.step(action, return_obs=False)
            if done:
                return reward * multiplier
            multiplier *= -1

        return 0
        ########################
    
    def pick_leaf(self) -> MCTSNode:
        # select the leaf node to expand
        # the leaf node is the node that has not been expanded
        # create and return a new node if game is not ended
        
        ########################
        # TODO: your code here #
        ########################
        curr = self.root
        while not curr.done:
            valid_actions = np.where(curr.action_mask == 1)[0]
            unexpanded_actions = [a for a in valid_actions if not curr.has_child(a)]
            if len(unexpanded_actions) > 0:
                action = np.random.choice(unexpanded_actions)
                return curr.add_child(action)
            else:
                action = self.uct_action_select(curr)
                curr = curr.get_child(action)
        return curr
        ########################
    
    def get_policy(self, node:MCTSNode = None) -> np.ndarray:
        # return the policy of the tree(root) after the search
        # the policy conmes from the visit count of each action 
        
        ########################
        # TODO: your code here #
        ########################
        if node is None:
            node = self.root
        sum_visits = np.sum(node.child_N_visit)
        if sum_visits == 0:
            valid_mask = node.action_mask == 1
            policy = np.zeros(node.n_action)
            policy[valid_mask] = 1.0 / np.sum(valid_mask)
            return policy
        return node.child_N_visit / sum_visits
        ########################

    def search(self):
        # search the tree for n_search times
        # eachtime, pick a leaf node, rollout the game (if game is not ended) 
        #   for n_rollout times, and backup the value.
        # return the policy of the tree after the search
        for _ in range(self.config.n_search):
            leaf = self.pick_leaf()
            value = 0
            if leaf.done:
                ########################
                # TODO: your code here #
                ########################
                value = -leaf.reward
                ########################
            else:
                ########################
                # TODO: your code here #
                ########################
                for _ in range(self.config.n_rollout):
                    value += self.rollout(leaf)
                value /= self.config.n_rollout
                ########################
            self.backup(leaf, value)

        return self.get_policy(self.root)