from tensorflow.keras import Model, layers

self.reshape = layers.Reshape([1, numFilters])

def h2c(hd,he):
    scores = tf.einsum('btf, btf-> bt', he, hd)
    weights = layers.Softmax(axis=1)(scores)
    contextVecter = tf.einsum('btf, bt-> bf', he, weights)
    return contextVecter

class ContextBlock(Model):
    def __init__(self,numFilters=32, attentionLen=48):
        super().__init__()
        self.he2hd = DecoderBlock(numFilters)
        self.reshape = layers.Reshape([1, numFilters])

    def call(self, he):
        numTime = he.shape[1]
        
        hd = self.he2hd(he,1)
        he0= self.reshape(he[:,0,:])
        contextVecter = self.reshape(h2c(hd,he0))
        if numTime is not None:
            for i in range(1,numTime):        
                hd = self.he2hd(he,i)
                #assert he.shape == hd.shape
                contextVecteri = self.reshape(h2c(hd,he[:,max(i-attentionLen,0):i,:]))
                contextVecter = tf.concat([contextVecter, contextVecteri], axis=1)
        return contextVecter #乘法 hd*he

        
