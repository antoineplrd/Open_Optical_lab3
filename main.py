# Lab 4 - Open Optical Network ANTOINE POUILLARD
import networkx as nx
from matplotlib import pyplot as plt
import string
import random
from Network import Network
from Connection import Connection


def main():
    network = Network()
    network.chanel_availability()

    """nodeValue = 'ABCDEF'
    signal_power = 0.001

    for i in range(0, 100):
        inputNode = random.choice(nodeValue)
        outputNode = random.choice(nodeValue)
        while inputNode == outputNode:  # if we have the same node
            inputNode = random.choice(nodeValue)
            outputNode = random.choice(nodeValue)
            if inputNode != outputNode:
                break

        connections = Connection(inputNode, outputNode, signal_power)

        network.stream(connections, 'latency')
        # network.stream(connections, 'snr')

    # network.draw() # modifier pour mettre le chemin ?"""


if "__main__" == __name__:
    main()
