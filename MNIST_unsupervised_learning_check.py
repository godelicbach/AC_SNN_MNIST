import bindsnet

from bindsnet.datasets import MNIST
from bindsnet.network import Network
from bindsnet.pipeline import Pipeline
from bindsnet.learning import MSTDPET
from bindsnet.encoding import rank_order, single
from bindsnet.network.topology import Connection
from bindsnet.environment import DatasetEnvironment, MNISTEnvironment
from bindsnet.network.nodes import Input, LIFNodes, NoisyLIFNodes
from bindsnet.pipeline.action import num_spike

N_INPUT = 784
N_HIDDEN = 512
N_ACTOR = 100
N_CRITIC = 100
THRESH = 1.0

W_MEAN = 0.0
W_STD = 0.5

DT = 0.01

# Build network.
network = Network(dt=DT)
# Layers of neurons.
input = Input(n=N_INPUT, shape=[N_INPUT], traces=True)
hidden = LIFNodes(n=N_HIDDEN, traces=True, reset=0.0, rest=0.0, thresh=THRESH, decay=0.01, lbound=-THRESH, refrac=0)


in_hidden = Connection(source=input, target=hidden, wmin=W_MEAN-W_STD, wmax=W_MEAN+W_STD)
hidden_lateral = Connection(source=hidden, target=hidden, wmin=-0.1, wmax=0)

# Add all layers and connections to the network.
network.add_layer(input, name='IN')
network.add_layer(hidden, name='HIDDEN')
network.add_connection(in_hidden, source='IN', target='HIDDEN')
network.add_connection(hidden_lateral, source='HIDDEN', target='HIDDEN')

environment = DatasetEnvironment(dataset=MNIST(path='../../data/MNIST', download=True), train=True)

pipeline = Pipeline(network, environment, encoding=rank_order,
                    time=10, plot_interval=1, plot_length=2.3, plot_type='line',
                    output='HIDDEN',)

# Run environment simulation and network training.
n_episode = 0
n_step = 0
while True:
    pipeline.step()
    n_step += 1
    if pipeline.done:
        n_episode += 1
        n_step = 0
        pipeline.reset_()
