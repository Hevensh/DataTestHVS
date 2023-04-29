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

class DecoderBlock(Model):
    def __init__(self,numFilters):
        super().__init__()
        self.decoder = DecoderLSTMcell(numFilters)
        self.reshape = layers.Reshape([1, numFilters])
        
    def call(self, inputs, ite):
        numTime = inputs.shape[1]
        
        h0 = self.reshape(inputs[:,ite,:])
        h, y = self.decoder(h0)        
        if numTime is not None:
            for i in range(1,min(NUM_ATTNTION_LEN,ite)):
                h, yi = self.decoder(h)
                y = tf.concat([y, yi], axis=1)

        return y
