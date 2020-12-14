

class Average(object):
    """ calculate average information for
    train process, include loss score and so on
    """
    def __init__(self, length=0):
        """ initial
        Args:
            length: length of data that need
                    calculate average value
        """
        self.item = 0
        self.history = 0
        self.value_num = 0
        self.length = length

    def update(self, value, num=1):
        """ update this function will clear history
        if length of history greater than self.length """
        if self.item > self.length:
            self.history = 0
            self.item = 0
            self.value_num = 0

        self.item += 1
        self.value_num += num
        self.history += value * num

    @property
    def average(self):
        """ calculate average of history """
        return self.history / self.value_num
