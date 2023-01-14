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
from Lightpath import Lightpath
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
        propagate = self._nodes.get(signal_information.path[0]).propagate(signal_information)
        self._route_space = self.chanel_availability()  # We update for each new path the avability
        return propagate

    def probe(self, signal_information):
        return self._nodes.get(signal_information.path[0]).probe(signal_information)

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
                signal_information = self.probe(Signal_information(0.001, j))
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

        Dataframe = self._weighted_paths
        Dataframe_occupancy = self._route_space
        all_paths = Dataframe['Paths'].tolist()
        all_noise_radio = Dataframe['Signal/noise (dB)'].tolist()
        all_paths_occupancy = Dataframe_occupancy['Paths'].tolist()
        all_occupancy = Dataframe_occupancy['availability'].tolist()
        noise_radio = min(all_noise_radio)
        result_path = list()
        free_chanel = {}
        Path_final = ""

        for path in all_paths:
            if path[0] == input_node and path[len(path) - 1] == output_node:
                test = True
                occupancy = all_occupancy[all_paths_occupancy.index(path)]
                final_occupancy = list(chain.from_iterable(occupancy))  # remove double [[]]
                for i in range(len(final_occupancy)):
                    if final_occupancy[i]:
                        test = True
                        free_chanel[path] = i  # faire un dictionnaire qui envoie le path avec le channel libre
                        break
                    else:
                        test = False

                if all_noise_radio[all_paths.index(path)] > noise_radio and test is True:
                    noise_radio = all_noise_radio[all_paths.index(path)]
                    Path_final = path

        channel = free_chanel.get(Path_final)  # get the channel value for the best snr path

        result_path.append(Path_final)
        result_path.append(channel)

        return result_path

    def find_best_latency(self, input_node, output_node):

        Dataframe = self._weighted_paths
        Dataframe_occupancy = self._route_space
        all_paths = Dataframe["Paths"].tolist()
        all_latency = Dataframe["Latency (s)"].tolist()
        all_paths_occupancy = Dataframe_occupancy['Paths'].tolist()
        all_occupancy = Dataframe_occupancy['availability'].tolist()
        latency = max(all_latency)
        result_path = list()
        free_chanel = {}
        Path_final = ""

        for path in all_paths:
            if path[0] == input_node and path[len(path) - 1] == output_node:
                test = True
                occupancy = all_occupancy[all_paths_occupancy.index(path)]
                final_occupancy = list(chain.from_iterable(occupancy))  # remove double [[]]
                for i in range(len(final_occupancy)):
                    if final_occupancy[i]:
                        test = True
                        free_chanel[path] = i  # faire un dictionnaire qui envoie le path avec le channel libre
                        break
                    else:
                        test = False

                if all_latency[all_paths.index(path)] < latency and test is True:
                    latency = all_latency[all_paths.index(path)]
                    Path_final = path

        channel = free_chanel.get(Path_final)  # get the channel value for the best snr path

        result_path.append(Path_final)
        result_path.append(channel)

        return result_path

    def stream(self, connection, label="latency"):

        input = connection.input
        output = connection.output
        signal_power = connection.signal_power

        if label == "snr":
            path_snr = self.find_best_snr(input, output)
            final_path_snr = path_snr[0]
            final_path_snr = list(final_path_snr.split("->"))
            freq_channel = path_snr[1]

            if final_path_snr != ['']:
                signal_information = Lightpath(freq_channel, signal_power, final_path_snr)
                propagate_snr = self.propagate(signal_information)
                connection.snr = propagate_snr.snr
            else:
                connection.snr = 0

        elif label == "latency":
            path_latency = self.find_best_latency(input, output)
            final_path_latency = path_latency[0]
            final_path_latency = list(final_path_latency.split("->"))
            freq_channel = path_latency[1]

            if final_path_latency != ['']:
                signal_information = Lightpath(freq_channel, signal_power, final_path_latency)
                propagate_latency = self.propagate(signal_information)
                connection.latency = propagate_latency.latency
            else:
                connection.latency = 'None'

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
                final_availability = list()

                for label in range(len(actualPath)):
                    if label < len(actualPath) - 1:  # dealing with the case for the last line
                        line = actualPath[label] + actualPath[label + 1]
                        availability_temp.append(self._lines.get(line).state)  # get the state for each line

                availability.append(availability_temp)  # send the state

                path_cpy = actualPath[:]
                path_cpy = '->'.join(path_cpy)
                self.probe(Signal_information(0.001, actualPath))
                test = list()  # we reset the list for each path
                for chemin in availability:  # for each path of each line
                    for index1 in range(len(chemin[0])):  # cross on different frequency (i.e. 10)
                        availability_path = list()  # reset to zero for each channel
                        for index2 in range(len(chemin)):  # cross on different lines of the path for same channel
                            availability_path.append(chemin[index2][index1])  # send each state with the same channel

                        if len(set(availability_path)) == 1 and availability_path[0] is True:  # only true
                            test.append(True)
                        elif len(set(availability_path)) == 1 and availability_path[0] is False:  # only false
                            test.append(False)
                        else:  # case with false and true
                            test.append(False)

                    final_availability.append(test)

                result_data.append(list([path_cpy, final_availability]))

        data = {

            "Paths": [i[0] for i in result_data],
            "availability": [i[1] for i in result_data],

        }
        df = pd.DataFrame(data)
        # print(tabulate(df, showindex=True, headers=df.columns))

        return df
