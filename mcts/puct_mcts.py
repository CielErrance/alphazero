from .node import MCTSNode, INF
from .config import MCTSConfig
from env.base_env import BaseGame

from model.linear_model_trainer import NumpyLinearModelTrainer
import numpy as np


class PUCTMCTS:
    def __init__(self, init_env:BaseGame, model: NumpyLinearModelTrainer, config: MCTSConfig, root:MCTSNode=None):
        self.model = model
        self.config = config
        self.root = root
        if root is None:
            self.init_tree(init_env)
        self.root.cut_parent()
    
    def init_tree(self, init_env:BaseGame):
        env = init_env.fork()
        obs = env.observation
        self.root = MCTSNode(
            action=None, env=env, reward=0
        )
        # compute and save predicted policy
        child_prior, _ = self.model.predict(env.compute_canonical_form_obs(obs, env.current_player))
        self.root.set_prior(child_prior)
    
    def get_subtree(self, action:int):
        # return a subtree with root as the child of the current root
        # the subtree represents the state after taking action
        if self.root.has_child(action):
            new_root = self.root.get_child(action)
            return PUCTMCTS(new_root.env, self.model, self.config, new_root)
        else:
            return None
    
    def puct_action_select(self, node:MCTSNode):
       # select the best action based on PUCB when expanding the tree
        
        ########################
        # TODO: your code here #
        ########################
        valid_mask = node.action_mask == 1 # 转换为bool数组
        N_parent = np.sum(node.child_N_visit)
        q = np.full(node.n_action, -INF, dtype=np.float64)
        valid_actions = np.where(valid_mask)[0]

        # 为根节点添加噪声
        if self.config.with_noise and node is self.root and not getattr(self, '_noise_applied', False):
            priors = node.child_priors.copy()
            noise = np.zeros_like(priors)
            if len(valid_actions) > 0:
                dir_noise = np.random.dirichlet([self.config.dir_alpha] * len(valid_actions))
                noise[valid_actions] = dir_noise
            node.child_priors = (1 - self.config.dir_epsilon) * priors + self.config.dir_epsilon * noise
            self._noise_applied = True

        # 统一按后继状态评估Q，若树中无子节点则临时模拟一步
        for action in valid_actions:
            if node.has_child(action):
                child = node.get_child(action)
                child_env = child.env
                reward = child.reward
                done = child_env.ended
            else:
                child_env = node.env.fork()
                _, reward, done = child_env.step(action)

            if done:
                q[action] = reward
                continue

            obs = child_env.observation
            _, child_value = self.model.predict(
                child_env.compute_canonical_form_obs(obs, child_env.current_player)
            )
            q[action] = -np.asarray(child_value).item()

        priors = node.child_priors
        explore = self.config.C * priors * np.sqrt(N_parent) / (1.0 + node.child_N_visit)
        pucb = q + explore
        pucb[~valid_mask] = -INF
        return np.argmax(pucb)

        ########################

    def backup(self, node:MCTSNode, value):
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
    
    def pick_leaf(self):
        # select the leaf node to expand
        # the leaf node is the node that has not been expanded
        # create and return a new node if game is not ended
        
        ########################
        # TODO: your code here #
        ########################
        curr = self.root
        while not curr.done:
            action = self.puct_action_select(curr)
            if curr.has_child(action):
                curr = curr.get_child(action)
                continue

            child = curr.add_child(action)
            if not child.done:
                obs = child.env.observation
                child_prior, _ = self.model.predict(
                    child.env.compute_canonical_form_obs(obs, child.env.current_player)
                )
                child.set_prior(child_prior)
            return child

        return curr
        ########################
    
    def get_policy(self, node:MCTSNode = None):
        # return the policy of the tree(root) after the search
        # the policy conmes from the visit count of each action 
        
        ########################
        # TODO: your code here #
        ########################
        if node is None:
            node = self.root

        visits = node.child_N_visit.astype(np.float64)
        valid_mask = node.action_mask == 1

        if np.sum(visits) == 0:
            policy = np.zeros(node.n_action, dtype=np.float64)
            policy[valid_mask] = 1.0 / np.sum(valid_mask)
            return policy

        temp = self.config.temperature

        visits = np.where(valid_mask, visits, 0.0)
        policy = visits ** (1.0 / temp)
        policy_sum = np.sum(policy)

        return policy / policy_sum
        ########################

    def search(self):
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
                # NOTE: you should compute the policy and value 
                #       using the value&policy model!
                obs = leaf.env.observation
                child_prior, value = self.model.predict(
                    leaf.env.compute_canonical_form_obs(obs, leaf.env.current_player)
                )
                leaf.set_prior(child_prior)
                value = np.asarray(value).item()
                ########################
            self.backup(leaf, value)
            
        return self.get_policy(self.root)