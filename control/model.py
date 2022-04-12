import torch
import torch.nn as nn
import torch.nn.functional as F

from distributions import Categorical, DiagGaussian
from utils import init, init_normc_


class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)


class BaseConv(nn.Module):
    def __init__(self, num_inputs):
        super(BaseConv, self).__init__()
        self.conv = nn.Sequential(nn.Conv2d(num_inputs, 64, 3, stride=2, padding=1),
                                  nn.ReLU(),
                                  nn.Conv2d(64, 64, 3, stride=2, padding=1),
                                  nn.ReLU(),
                                  nn.Conv2d(64, 64, 3, stride=2, padding=1),
                                  nn.ReLU(),
                                  nn.Conv2d(64, 64, 3, stride=2, padding=1),
                                  nn.ReLU(),
                                  nn.Conv2d(64, 64, 3, stride=2, padding=1),
                                  nn.ReLU()
                                  )

    def forward(self, x):
        return self.conv(x)


class ICM(torch.nn.Module):
    def __init__(self, action_space, state_size, env_name, num_inputs=1, cnn_head=True, base_kwargs=None):
        super(ICM, self).__init__()
        if cnn_head:
            self.head = BaseConv(num_inputs)

        if action_space.__class__.__name__ == "Discrete":
            action_space = action_space.n
            self.state_size = 64 * 3 * 3
        else:
            raise NotImplementedError
            # TODO: for continuous action space, loss is gaussian mean or output?
            action_space = action_space.shape[0]

        self.forward_model = nn.Sequential(
            nn.Linear(state_size + action_space, 1024),
            nn.ReLU(),
            nn.Linear(1024, state_size))

        self.inverse_model = nn.Sequential(
            nn.Linear(state_size * 2, 1024),
            nn.ReLU(),
            nn.Linear(1024, action_space),
            nn.ReLU())
        self._initialize_weights()

    def _initialize_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Conv2d) or isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                # nn.init.kaiming_uniform_(module.weight)
                nn.init.constant_(module.bias, 0)

    def forward(self, state, next_state, action):
        if hasattr(self, 'head'):
            phi1 = self.head(state)
            phi2 = self.head(next_state)
        else:
            phi1 = state
            phi2 = next_state
        phi1_local = phi1.detach().view(-1, self.state_size)
        phi2_local = phi2.detach().view(-1, self.state_size)
        # forward model: f(phi1,asample) -> phi2
        phi2_pred = self.forward_model(torch.cat([phi1_local, action], 1))

        # inverse model: g(phi1,phi2) -> a_inv: [None, ac_space]
        action_pred = F.softmax(self.inverse_model(torch.cat([phi1_local, phi2_local], 1)), -1)
        return action_pred, phi2_pred, phi2_local


class Policy(nn.Module):
    def __init__(self, obs_shape, action_space, env_name, base_kwargs=None):
        super(Policy, self).__init__()
        if base_kwargs is None:
            base_kwargs = {}

        if len(obs_shape) == 3:
            self.base = CNNBase(obs_shape[0], env_name, **base_kwargs)
        elif len(obs_shape) == 1:
            self.base = MLPBase(obs_shape[0], **base_kwargs)
        else:
            raise NotImplementedError

        if action_space.__class__.__name__ == "Discrete":
            num_outputs = action_space.n
            self.dist = Categorical(self.base.output_size, num_outputs)
            self.num_outputs = num_outputs
            self.discrete = True
        elif action_space.__class__.__name__ == "Box":
            num_outputs = action_space.shape[0]
            self.num_outputs = num_outputs
            self.discrete = False
            self.dist = DiagGaussian(self.base.output_size, num_outputs)
        else:
            raise NotImplementedError

    @property
    def is_recurrent(self):
        return self.base.is_recurrent

    @property
    def recurrent_hidden_state_size(self):
        """Size of rnn_hx."""
        return self.base.recurrent_hidden_state_size

    def forward(self, inputs, rnn_hxs, masks):
        raise NotImplementedError

    def act(self, inputs, rnn_hxs, masks, deterministic=False):
        value, actor_features, rnn_hxs = self.base(inputs, rnn_hxs, masks)
        dist = self.dist(actor_features)

        if deterministic:
            action = dist.mode()
        else:
            action = dist.sample()

        action_log_probs = dist.log_probs(action)
        dist_entropy = dist.entropy().mean()
        # import pdb;pdb.set_trace()
        return value, action, action_log_probs, rnn_hxs, actor_features

    def get_value(self, inputs, rnn_hxs, masks):
        value, _, _ = self.base(inputs, rnn_hxs, masks)
        return value

    def evaluate_actions(self, inputs, rnn_hxs, masks, action):
        value, actor_features, rnn_hxs = self.base(inputs, rnn_hxs, masks)
        dist = self.dist(actor_features)

        action_log_probs = dist.log_probs(action)
        dist_entropy = dist.entropy().mean()

        return value, action_log_probs, dist_entropy, rnn_hxs


class ICM_Policy(Policy):
    def __init__(self, obs_shape, action_space, env_name, base_kwargs=None):
        super(ICM_Policy, self).__init__(obs_shape, action_space, env_name, base_kwargs)
        if len(obs_shape) == 3:
            self.icm = ICM(action_space=action_space, state_size=576, num_inputs=obs_shape[0],
                           env_name=env_name, cnn_head=True, base_kwargs=base_kwargs)
        elif len(obs_shape) == 1:
            self.icm = ICM(action_space=action_space, state_size=obs_shape[0], num_inputs=1,
                           env_name=env_name, cnn_head=False, base_kwargs=base_kwargs)
        else:
            raise NotImplementedError

    def act(self, inputs, rnn_hxs, masks, deterministic=False):
        value, actor_features, rnn_hxs = self.base(inputs, rnn_hxs, masks)
        dist = self.dist(actor_features)

        if deterministic:
            action = dist.mode()
        else:
            action = dist.sample()

        action_log_probs = dist.log_probs(action)
        dist_entropy = dist.entropy().mean()
        action_prob = torch.exp(action_log_probs)
        # import pdb;pdb.set_trace()
        return value, action, action_log_probs, rnn_hxs, actor_features, action_prob

    def get_icm_loss(self, states, next_states, action, device):
        action_oh = action
        if self.discrete:
            action_oh = torch.zeros((1, self.num_outputs))
            action_oh[0, action.view(-1)] = 1
            if torch.cuda.is_available():
                action_oh.to('cuda:0')
        print(action_oh.is_cuda)
        action_pred, phi2_pred, phi2 = self.icm(states, next_states, action_oh)
        inverse_loss = F.cross_entropy(action_pred, action_oh)
        forward_loss = 0.5 * F.mse_loss(phi2_pred, phi2)
        return inverse_loss, forward_loss


class NNBase(nn.Module):

    def __init__(self, recurrent, recurrent_input_size, hidden_size):
        super(NNBase, self).__init__()

        self._hidden_size = hidden_size
        self._recurrent = recurrent

        if recurrent:
            self.gru = nn.GRUCell(recurrent_input_size, hidden_size)
            nn.init.orthogonal_(self.gru.weight_ih.data)
            nn.init.orthogonal_(self.gru.weight_hh.data)
            self.gru.bias_ih.data.fill_(0)
            self.gru.bias_hh.data.fill_(0)

    @property
    def is_recurrent(self):
        return self._recurrent

    @property
    def recurrent_hidden_state_size(self):
        if self._recurrent:
            return self._hidden_size
        return 1

    @property
    def output_size(self):
        return self._hidden_size

    def _forward_gru(self, x, hxs, masks):
        if x.size(0) == hxs.size(0):
            x = hxs = self.gru(x, hxs * masks)
        else:
            # x is a (T, N, -1) tensor that has been flatten to (T * N, -1)
            N = hxs.size(0)
            T = int(x.size(0) / N)

            # unflatten
            x = x.view(T, N, x.size(1))

            # Same deal with masks
            masks = masks.view(T, N, 1)

            outputs = []
            for i in range(T):
                hx = hxs = self.gru(x[i], hxs * masks[i])
                outputs.append(hx)

            # assert len(outputs) == T
            # x is a (T, N, -1) tensor
            x = torch.stack(outputs, dim=0)
            # flatten
            x = x.view(T * N, -1)

        return x, hxs


class Print(nn.Module):
    def __init__(self):
        super(Print, self).__init__()

    def forward(self, x):
        print('layer input:', x.shape)
        return x


class CNNBase(NNBase):
    def __init__(self, num_inputs, env_name, recurrent=False, hidden_size=128):
        super(CNNBase, self).__init__(recurrent, hidden_size, hidden_size)

        init_ = lambda m: init(m,
                               nn.init.orthogonal_,
                               lambda x: nn.init.constant_(x, 0),
                               nn.init.calculate_gain('relu'))

        if "MiniWorld" in env_name:
            finalsize = 32 * 6 * 4
        else:
            finalsize = 32 * 7 * 7
        self.main = nn.Sequential(
            init_(nn.Conv2d(num_inputs, 32, 8, stride=4)), nn.ReLU(),
            init_(nn.Conv2d(32, 64, 4, stride=2)), nn.ReLU(),
            init_(nn.Conv2d(64, 32, 3, stride=1)), nn.ReLU(), Flatten(),
            #            init_(nn.Linear(32 * 6 * 4, hidden_size)), nn.ReLU()
            init_(nn.Linear(finalsize, hidden_size)), nn.ReLU()
        )

        init_ = lambda m: init(m,
                               nn.init.orthogonal_,
                               lambda x: nn.init.constant_(x, 0))

        self.critic_linear = init_(nn.Linear(hidden_size, 1))

        self.train()

    def forward(self, inputs, rnn_hxs, masks):
        # print(inputs.size())

        x = inputs / 255.0
        # print(x.size())

        x = self.main(x)
        # import pdb;pdb.set_trace()
        # print(x.size())

        if self.is_recurrent:
            x, rnn_hxs = self._forward_gru(x, rnn_hxs, masks)

        return self.critic_linear(x), x, rnn_hxs


class MLPBase(NNBase):
    def __init__(self, num_inputs, recurrent=False, hidden_size=64):
        super(MLPBase, self).__init__(recurrent, num_inputs, hidden_size)

        if recurrent:
            num_inputs = hidden_size

        init_ = lambda m: init(m,
                               init_normc_,
                               lambda x: nn.init.constant_(x, 0))

        self.actor = nn.Sequential(
            init_(nn.Linear(num_inputs, hidden_size)),
            nn.Tanh(),
            init_(nn.Linear(hidden_size, hidden_size)),
            nn.Tanh()
        )

        self.critic = nn.Sequential(
            init_(nn.Linear(num_inputs, hidden_size)),
            nn.Tanh(),
            init_(nn.Linear(hidden_size, hidden_size)),
            nn.Tanh()
        )

        self.critic_linear = init_(nn.Linear(hidden_size, 1))

        self.train()

    def forward(self, inputs, rnn_hxs, masks):
        x = inputs

        if self.is_recurrent:
            x, rnn_hxs = self._forward_gru(x, rnn_hxs, masks)

        hidden_critic = self.critic(x)
        hidden_actor = self.actor(x)

        return self.critic_linear(hidden_critic), hidden_actor, rnn_hxs
