import bindsnet

from bindsnet.datasets import MNIST
from bindsnet.network import Network
from bindsnet.pipeline import Pipeline
from bindsnet.learning import MSTDPET
from bindsnet.encoding import rank_order, single
from bindsnet.network.topology import Connection
from bindsnet.datasets import CustomDataset
from bindsnet.environment import DatasetEnvironment, MNISTEnvironment
from bindsnet.network.nodes import Input, LIFNodes, NoisyLIFNodes
from bindsnet.pipeline.action import select_max
from bindsnet.agent import Agent

import numpy as np

N_INPUT = 100
N_OUTPUT = 100
THRESH = 1.0

W_MEAN = 0.01
W_STD = 0.5

TIME=10
DT = 1.0

N_DATA = 10
N_FEATURE = N_INPUT

#Make custom dataset
def generate_data(num_samples, num_features):
    data = np.random.rand(num_samples, num_features)
    label = np.arange(num_samples)
    return data, label

train_data, train_label = generate_data(N_DATA, N_FEATURE)

custom_dataset = CustomDataset(train_data=train_data, train_label=train_label)

#################
# Build network.
#################
network = Network(dt=DT)
# Layers of neurons.
input = Input(n=N_INPUT, shape=[N_INPUT], traces=True)
output = NoisyLIFNodes(n=N_OUTPUT, traces=True, reset=0.0, rest=0.0, thresh=THRESH,
    decay=0.01, lbound=-THRESH, refrac=0, noise_std=0.3, noise_decay=0.0)
in_out = Connection(source=input, target=output, wmin=W_MEAN-W_STD,
    wmax=W_MEAN+W_STD, update_rule=MSTDPET, dt=DT, decay=0.1, nu=1.0)

# Add all layers and connections to the network.
network.add_layer(input, name='IN')
network.add_layer(output, name='OUT')
network.add_connection(in_out, source='IN', target='OUT')

##################
# Construct agent.
##################
agent = Agent(network, dt=DT, time=TIME, epsilon=0.0, epsilon_decay=0.97,
    encoding=single, num_action = N_DATA, action_function=select_max,
    output_name='OUT')

###################
# Make Environment.
###################
environment = DatasetEnvironment(dataset=custom_dataset)

################################
# Merge everything into Pipeline
################################
pipeline = Pipeline(agent, environment, plot_interval=1, plot_length=1.0,
                    plot_type='line')

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
