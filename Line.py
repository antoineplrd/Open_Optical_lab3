class Line:

    def __init__(self, label, length, number_of_channel=10):
        self._label = label
        self._length = length
        self._successive = dict()
        self._number_of_channel = number_of_channel
        self._state = [True] * number_of_channel

    @property
    def label(self):
        return self._label

    @property
    def length(self):
        return self._length

    @property
    def successive(self):
        return self._successive

    @label.setter
    def label(self, value):
        self._label = value

    @length.setter
    def length(self, length):
        self._length = length

    @successive.setter
    def successive(self, value):
        self._successive = value

    @property
    def state(self):
        return self._state

    @state.setter
    def state(self, state):
        self._state = state

    def latency_generation(self):
        result = self._length / ((2 / 3) * 299792458)
        return result

    def noise_generation(self, signal_power):
        return pow(10, -9) * signal_power * self._length

    def propagate(self, signal_information): # add lightpath
        #channel = lightpath.channel
        #self.state[channel] = False
        signal_information.update_noise_power(self.noise_generation(signal_information.signal_power))
        signal_information.UpdateLatency(self.latency_generation())
        return self._successive.get(signal_information.path[0]).propagate(signal_information)

    def probe(self, signal_information):
        signal_information.update_noise_power(self.noise_generation(signal_information.signal_power))
        signal_information.UpdateLatency(self.latency_generation())
        return self._successive.get(signal_information.path[0]).probe(signal_information)
