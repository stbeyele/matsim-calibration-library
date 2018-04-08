from abc import ABCMeta, abstractmethod

class TestABC(object):
    __metaclass__ = ABCMeta

    @abstractmethod
    def input(self, input):
        """test"""
        return

    @abstractmethod
    def save(self, output, data):
        """Test"""
        return

    @abstractpropperty
    def value(self):
        return 'Test'


class SubclassImplementation(object):

    def load(self, input):
        base_data = super(TestABC, self).retrieve_values(input)
        return input.read()

    def save(self, output, data):
        return output.write(data)

    @property
    def value(self):
        return 'concrete property'


TestABC.register(SubclassImplementation)



