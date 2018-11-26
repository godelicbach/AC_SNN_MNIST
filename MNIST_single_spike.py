import bindsnet

import math
from bindsnet.datasets import MNIST
from bindsnet.network import Network
from bindsnet.pipeline import Pipeline
from bindsnet.learning import MSTDPET
from bindsnet.encoding import rank_order, single
from bindsnet.network.topology import Connection
from bindsnet.environment import DatasetEnvironment
from bindsnet.network.nodes import Input, LIFNodes
from bindsnet.pipeline.action import select_multinomial

N_INPUT = 784
N_HIDDEN = 50
N_ACTOR = 100
N_CRITIC = 10
THRESH = 1.0

W_MEAN = 0.0
W_STD = 0.5

# Build network.
network = Network(dt=0.5)

# Layers of neurons.
input = Input(n=N_INPUT, shape=[N_INPUT], traces=True)
hidden = LIFNodes(n=N_HIDDEN, traces=True, reset=0.0, rest=0.0, thresh=THRESH, decay=0.01, lbound=-THRESH, refrac=0)
actor = LIFNodes(n=N_ACTOR, refrac=0, traces=True, reset=0.0, rest=0.0, thresh=THRESH, decay=0.01, lbound=-THRESH, noise_std = 1.0)
critic = LIFNodes(n=N_CRITIC, refrac=0, traces=True, reset=0.0, rest=0.0, thresh=THRESH, decay=0.01, lbound=-THRESH, noise_std = 0)


in_hidden = Connection(source=input, target=hidden, wmin=W_MEAN-W_STD, wmax=W_MEAN+W_STD)
hidden_lateral = Connection(source=hidden, target=hidden, wmin=-0.5, wmax=0)
hidden_actor = Connection(source=hidden, target=actor, wmin=W_MEAN-W_STD, wmax=W_MEAN+W_STD)
actor_lateral = Connection(source=actor, target=actor, wmin=-0.5, wmax=0)
hidden_critic = Connection(source=hidden, target=critic, wmin=W_MEAN-W_STD, wmax=W_MEAN+W_STD)

# Add all layers and connections to the network.
network.add_layer(input, name='IN')
network.add_layer(hidden, name='HIDDEN')
network.add_layer(actor, name='ACTOR')
network.add_layer(critic, name='CRITIC')
network.add_connection(in_hidden, source='IN', target='HIDDEN')
network.add_connection(hidden_lateral, source='HIDDEN', target='HIDDEN')
network.add_connection(hidden_actor, source='HIDDEN', target='ACTOR')
network.add_connection(actor_lateral, source='ACTOR', target='ACTOR')
network.add_connection(hidden_critic, source='HIDDEN', target='CRITIC')

environment = DatasetEnvironment(dataset=MNIST(path='../../data/MNIST',
                                 download=True), train=True)

pipeline = Pipeline(network, environment, encoding=rank_order,
                    time=100, plot_interval=1, plot_length=2.3, plot_type='line',
                    output='ACTOR')

# Run environment simulation and network training.
n_episode = 0
n_step = 0
while True:
    pipeline.step()
    n_step += 1
    print('{}th episode\'s {}th step'.format(n_episode,n_step))
    if pipeline.done:
        n_episode += 1
        n_step = 0
        pipeline.reset_()
