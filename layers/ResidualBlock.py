from tensorflow.keras import Model, layers

class ResidualBlock(Model):
    def __init__(self,numFilters=32,sizeFilters=3):
        super().__init__()
        self.conv1 = layers.Conv1D(numFilters, sizeFilters, activation='relu', padding='causal')
        self.conv2 = layers.Conv1D(numFilters, sizeFilters, activation='relu', padding='causal')
        
    def call(self, inputs):
        x1 = self.conv1(inputs)
        x2 = self.conv2(x1)
        return x2 + inputs
