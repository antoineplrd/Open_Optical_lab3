from Signal_information import Signal_information


class Lightpath(Signal_information):
    def __init__(self, channel, signal_power, path):
        super().__init__(signal_power, path)
        self._channel = channel

    @property
    def channel(self):
        return self._channel

    @channel.setter
    def channel(self, value):
        self._channel = value
