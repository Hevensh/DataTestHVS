from tensorflow.keras import Model, layers

class DecoderLSTMcell(Model):
    def __init__(self,numFilters):
        super().__init__()
        self.fc1 = layers.Dense(numFilters, activation='sigmoid', use_bias=False)
        self.fc2 = layers.Dense(numFilters, activation='sigmoid', use_bias=False)
        
    def call(self, inputs):
        h = self.fc1(inputs)
        y = self.fc2(h)
        return h, y
