import numpy as np

from .base_policy import BasePolicy


class MPCPolicy(BasePolicy):

    def __init__(self,
                 env,
                 ac_dim,
                 dyn_models,
                 horizon,
                 N,
                 sample_strategy='random',
                 **kwargs
                 ):
        super().__init__(**kwargs)

        # init vars
        self.env = env
        self.dyn_models = dyn_models
        self.horizon = horizon
        self.N = N
        self.data_statistics = None  # NOTE must be updated from elsewhere

        self.ob_dim = self.env.observation_space.shape[0]

        # action space
        self.ac_space = self.env.action_space
        self.ac_dim = ac_dim
        self.low = self.ac_space.low
        self.high = self.ac_space.high

        # Sampling strategy
        allowed_sampling = ('random',)
        assert sample_strategy in allowed_sampling, f"sample_strategy must be one of the following: {allowed_sampling}"
        self.sample_strategy = sample_strategy

        print(f"Using action sampling strategy: {self.sample_strategy}")
    def sample_action_sequences(self, num_sequences, horizon, obs=None):
        return np.random.uniform(low=self.low, high=self.high, size=(num_sequences, horizon, self.ac_dim))

    def evaluate_candidate_sequences(self, candidate_action_sequences, obs):
        rewards = np.zeros(len(candidate_action_sequences))
        for model in self.dyn_models: 
            reward = self.calculate_sum_of_rewards(obs, candidate_action_sequences, model)
            rewards += reward
        return rewards / len(self.dyn_models)
            

    def get_action(self, obs):
        if self.data_statistics is None:
            return self.sample_action_sequences(num_sequences=1, horizon=1)[0]

        # sample random actions (N x horizon x action_dim)
        candidate_action_sequences = self.sample_action_sequences(
            num_sequences=self.N, horizon=self.horizon, obs=obs)

        predicted_rewards = self.evaluate_candidate_sequences(candidate_action_sequences, obs)

        # pick the action sequence and return the 1st element of that sequence
        best_action_sequence = candidate_action_sequences[np.argmax(predicted_rewards)]
        action_to_take =  best_action_sequence[0]
        return action_to_take[None]  # Unsqueeze the first index

    def calculate_sum_of_rewards(self, obs, candidate_action_sequences, model):
        """

        :param obs: numpy array with the current observation. Shape [D_obs]
        :param candidate_action_sequences: numpy array with the candidate action
        sequences. Shape [N, H, D_action] where
            - N is the number of action sequences considered
            - H is the horizon
            - D_action is the action of the dimension
        :param model: The current dynamics model.
        :return: numpy array with the sum of rewards for each action sequence.
        The array should have shape [N].
        """
        (N, H, D_action) = candidate_action_sequences.shape
        sum_of_rewards = np.zeros(N)  # TODO (Q2)
        # For each candidate action sequence, predict a sequence of
        # states for each dynamics model in your ensemble.
        # Once you have a sequence of predicted states from each model in
        # your ensemble, calculate the sum of rewards for each sequence
        # using `self.env.get_reward(predicted_obs, action)`
        # You should sum across `self.horizon` time step.
        # Hint: you should use model.get_prediction and you shouldn't need
        #       to import pytorch in this file.
        # Hint: Remember that the model can process observations and actions
        #       in batch, which can be much faster than looping through each
        #       action sequence.

        cur_obs = np.repeat(obs[None, :], N, axis=0)
        for h in range(H):
            pred = model.get_prediction(cur_obs, candidate_action_sequences[:, h, :], self.data_statistics)
            reward, _ = self.env.get_reward(pred, candidate_action_sequences[:, h, :])
            sum_of_rewards += reward
            cur_obs = pred

        return sum_of_rewards
