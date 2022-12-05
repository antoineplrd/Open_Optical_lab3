import Signal_information


class Lightpath(Signal_information):
    def __init__(self, channel):
        super.__init__(self)
        self._channel = channel

    @property
    def channel(self):
        return self._channel

    @channel.setter
    def channel(self, value):
        self._channel = value
