class SimpleClassForGreeting:
    """Class for greeting"""

    def __init__(self, name):
        self.name = name

    def greet(self):
        print("Hello, my name is " + self.name)
