import tensorflow as tf 
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
    def __init__(self,numFilters,attentionLen):
        super().__init__()
        self.decoder = DecoderLSTMcell(numFilters)
        self.reshape = layers.Reshape([1, numFilters])
        self.attentionLen = attentionLen
        
    def call(self, inputs, ite):
        numTime = inputs.shape[1]
        
        h0 = self.reshape(inputs[:,ite,:])
        h, y = self.decoder(h0)        
        if numTime is not None:
            for i in range(1,min(self.attentionLen,ite)):
                h, yi = self.decoder(h)
                y = tf.concat([y, yi], axis=1)

        return y

def h2c(hd,he):
    scores = tf.einsum('btf, btf-> bt', he, hd)
    weights = layers.Softmax(axis=1)(scores)
    contextVecter = tf.einsum('btf, bt-> bf', he, weights)
    return contextVecter

class ContextBlock(Model):
    def __init__(self,numFilters=32, attentionLen=48):
        super().__init__()
        self.he2hd = DecoderBlock(numFilters,attentionLen)
        self.reshape = layers.Reshape([1, numFilters])
        self.attentionLen = attentionLen

    def call(self, he):
        numTime = he.shape[1]
        
        hd = self.he2hd(he,1)
        he0= self.reshape(he[:,0,:])
        contextVecter = self.reshape(h2c(hd,he0))
        if numTime is not None:
            for i in range(1,numTime):        
                hd = self.he2hd(he,i)
                #assert he.shape == hd.shape
                contextVecteri = self.reshape(h2c(hd,he[:,max(i-self.attentionLen,0):i,:]))
                contextVecter = tf.concat([contextVecter, contextVecteri], axis=1)
        return contextVecter
