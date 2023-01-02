import json
import math
from math import *
import pandas as pd
import networkx as nx
from matplotlib import pyplot as plt
from Node import Node
from Connection import Connection
from Line import Line
from Signal_information import Signal_information
from tabulate import tabulate
from itertools import chain
import numpy as np


class Network:
    def __init__(self):
        self._nodes = dict()
        self._lines = dict()

        open_Json = open("nodes.json", "r")
        data = json.loads(open_Json.read())

        for i in data:
            label = i
            connected_nodes = list()
            connected_lines = list()

            for k in data[i]["position"]:
                connected_lines.append(k)
            for j in data[i]["connected_nodes"]:
                # implementation for line between 2 nodes: AB, BA
                connected_nodes.append(j)
            node = Node(label, (connected_lines[0], connected_lines[1]), connected_nodes)
            self._nodes.update({label: node})

        open_Json.close()

        for i in self._nodes:
            # length between 2 nodes
            labelX = self._nodes.get(i).position[0]
            labelY = self._nodes.get(i).position[1]
            for j in self._nodes.get(i).connected_nodes:
                label_lines = i + j
                nextNodeX = self._nodes.get(j).position[0]
                nextNodeY = self._nodes.get(j).position[1]
                Distance_lines = sqrt((nextNodeX - labelX) ** 2 + (nextNodeY - labelY) ** 2)

                line = Line(label_lines, Distance_lines)
                self._lines.update({label_lines: line})

        self.connect()
        self._weighted_paths = self.ex5()
        self._route_space = self.chanel_availability()

    @property
    def nodes(self):
        return self._nodes

    @nodes.setter
    def nodes(self, nodes):
        self._nodes = nodes

    @property
    def lines(self):
        return self._lines

    @lines.setter
    def lines(self, lines):
        self._lines = lines

    def connect(self):
        for i in self._lines:
            successive_nodes = dict()
            successive_nodes.update({i[0]: self._nodes.get(i[0])})
            successive_nodes.update({i[1]: self._nodes.get(i[1])})
            self.lines.get(i).successive = successive_nodes
        for i in self._nodes:
            successive_lines = dict()
            for j in self._nodes.get(i).connected_nodes:
                successive_lines.update({i + j: self._lines.get(i + j)})
            self._nodes.get(i).successive = successive_lines

    def find_paths(self, start_node, end_node, path=None):

        if path is None:
            path = []
        graph_dict = self._nodes.get(start_node).connected_nodes
        path = path + [start_node]
        if start_node == end_node:
            return [path]
        paths = []
        for actual_node in graph_dict:
            if actual_node not in path:
                extended_paths = self.find_paths(actual_node, end_node, path)
                for p in extended_paths:
                    paths.append(p)
        return paths

    def propagate(self, signal_information):
        return self._nodes.get(signal_information.path[0]).propagate(signal_information)

    def draw(self):
        G = nx.Graph()
        for i in self._nodes:
            G.add_nodes_from(self._nodes.get(i).label, pos=(self._nodes.get(i).position[0],
                                                            self._nodes.get(i).position[1]))
        for j in self._lines:
            G.add_edges_from([(j[0], j[1])])
        pos = nx.get_node_attributes(G, 'pos')
        nx.draw(G, pos, with_labels=True)
        plt.show()

    def ex5(self):
        var = list()
        path = list()
        for i in self._nodes:
            for j in self._nodes:
                if not (var.__contains__((i, j)) or i == j):
                    var.append((i, j))
                    path.append(self.find_paths(i, j))

        test = list()

        for i in path:
            for j in i:
                path_cpy = j[:]
                path_cpy = '->'.join(path_cpy)
                signal_information = self.propagate(Signal_information(0.001, j))
                ratio = 10 * math.log10(signal_information.signal_power / signal_information.noise_power)
                test.append(list([path_cpy, signal_information.latency, signal_information.noise_power, ratio]))

        data = {

            "Paths": [i[0] for i in test],
            "Latency (s)": [i[1] for i in test],
            "Noise power (W)": [i[2] for i in test],
            "Signal/noise (dB)": [i[3] for i in test]
        }
        df = pd.DataFrame(data)

        return df

    def find_best_snr(self, input_node, output_node):
        print("salut")
        global result
        best_path = ""
        temp_path = ""
        temp = 0
        full_paths = list()
        Dataframe = self._weighted_paths
        Dataframe_occupancy = self._route_space
        all_paths = Dataframe['Paths'].tolist()
        all_paths_occupancy = Dataframe_occupancy['Paths'].tolist()
        all_occupancy = Dataframe_occupancy['availability'].tolist()
        all_noise_radio = Dataframe['Signal/noise (dB)'].tolist()
        noise_radio = min(all_noise_radio)

        while True:
            for path in all_paths:
                if path[0] == input_node and path[len(path) - 1] == output_node:
                    if all_noise_radio[all_paths.index(path)] > noise_radio and path not in full_paths:
                        noise_radio = all_noise_radio[all_paths.index(path)]
                        temp_path = path

            for path_occupancy in all_paths_occupancy:
                if temp_path == path_occupancy and temp_path not in full_paths:  # case where we have not already check this path
                    occupancy = all_occupancy[all_paths_occupancy.index(path_occupancy)]

                    final_occupancy = list(chain.from_iterable(occupancy))  # remove double [[]]

                    if True in final_occupancy:
                        result = True
                    else:
                        result = False

            if result:
                best_path = temp_path
                temp = 1

            elif not result and len(full_paths) == len(all_paths_occupancy):
                best_path = ""
                temp = 1

            elif not result:
                full_paths.append(temp_path)

            if temp == 1:
                break

        return best_path

    def find_best_latency(self, input_node, output_node):
        global result
        best_path = ""
        temp_path = ""
        temp = 0
        full_paths = list()
        Dataframe = self._weighted_paths
        Dataframe_occupancy = self._route_space
        all_paths = Dataframe["Paths"].tolist()
        all_paths_occupancy = Dataframe_occupancy['Paths'].tolist()
        all_occupancy = Dataframe_occupancy['availability'].tolist()
        all_latency = Dataframe["Latency (s)"].tolist()
        latency = max(all_latency)

        while True:
            for path in all_paths:

                if path[0] == input_node and path[len(path) - 1] == output_node:
                    if all_latency[all_paths.index(path)] < latency and path not in full_paths:
                        latency = all_latency[all_paths.index(path)]
                        temp_path = path

            for path_occupancy in all_paths_occupancy:
                if temp_path == path_occupancy and temp_path not in full_paths:  # case where we have not already
                    # check this path
                    occupancy = all_occupancy[all_paths_occupancy.index(path_occupancy)]

                    final_occupancy = list(chain.from_iterable(occupancy))  # remove double [[]]

                    if True in final_occupancy:
                        result = True
                    else:
                        result = False

            if result is True:

                best_path = temp_path
                temp = 1

            elif not result and len(full_paths) == len(all_paths_occupancy):
                best_path = ""
                temp = 1

            elif not result:
                full_paths.append(temp_path)

            if temp == 1:
                break

        return best_path

    def stream(self, connection, label="latency"):

        input = connection.input
        output = connection.output
        signal_power = connection.signal_power

        if label == "snr":
            path_snr = self.find_best_latency(input, output)
            path_snr = list(path_snr.split("->"))

            signal_information = Signal_information(signal_power, path_snr)
            if path_snr != "":
                """availability = self._route_space[self._route_space['Paths'] == path_snr]['availability'].iloc[0]
                if 'Occupied' in availability:
                    print(f'Cannot stream signal on path {path_snr}. One or more channels are occupied.')
                else:
                    print(f'Streaming signal on path {path_snr}.')"""

                propagate_snr = self.propagate(signal_information)
                connection.snr = propagate_snr.snr
            else:
                connection.snr = 0

        elif label == "latency":
            path_latency = self.find_best_latency(input, output)
            path_latency = list(path_latency.split("->"))

            signal_information = Signal_information(signal_power, path_latency)
            if path_latency != "":
                propagate_latency = self.propagate(signal_information)
                connection.latency = propagate_latency.latency
            else:
                connection.latency = 'None'

    def probe(self, signal_information):
        return self._nodes.get(signal_information.path[0]).probe(signal_information)

    def chanel_availability(self):
        var = list()
        path = list()

        for i in self._nodes:
            for j in self._nodes:
                if not (var.__contains__((i, j)) or i == j):
                    var.append((i, j))
                    path.append(self.find_paths(i, j))

        result_data = list()

        for AllPaths in path:
            for actualPath in AllPaths:
                availability_temp = list()
                availability = list()
                availability_path = list()
                test = list()
                final_availability = list()

                for label in range(len(actualPath)):
                    if label < len(actualPath) - 1:  # dealing with the case for the last line
                        line = actualPath[label] + actualPath[label + 1]
                        availability_temp.append(self._lines.get(line).state)

                availability.append(availability_temp)

                path_cpy = actualPath[:]
                path_cpy = '->'.join(path_cpy)
                self.probe(Signal_information(0.001, actualPath))
                for chemin in availability:

                    for index1 in range(len(chemin[0])):
                        for index2 in range(len(chemin)):
                            availability_path.append(chemin[index2][index1])

                        if len(set(availability_path)) == 1 and availability_path[0] is True:
                            test.append(True)
                        elif len(set(availability_path)) == 1 and test[0] is False:
                            test.append(False)
                        else:
                            test.append(False)
                            availability_path = []
                    final_availability.append(test)

                result_data.append(list([path_cpy, final_availability]))

        data = {

            "Paths": [i[0] for i in result_data],
            "availability": [i[1] for i in result_data],

        }
        df = pd.DataFrame(data)
        #print(tabulate(df, showindex=True, headers=df.columns))

        return df
