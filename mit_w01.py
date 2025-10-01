#tensorflow implementation of a dense layer
class MyDenseLayer(tf.keras.layers.Layer):
    def __init__(self, input_dim,output_dim):
        super(MyDenseLayer, self).__init__()
        #initialize weights and biases
        self.w = self.add_weight([input_dim, output_dim])
        self.b = self.add_weight([1,output_dim])
    def call(self, inputs):
        z = tf.matmul(inputs, self.w) + self.b #forward propagation
        output = tf.math.sigmoid(z) #activation function
        return output

#simplified call to a dense layer using tensorflow
import tensorflow as tf
layer = tf.keras.layers.Dense(units=2)

#multiple output perceptron using tensorflow
import tensorflow as tf
model = tf.keras.Sequential([
    tf.keras.layers.Dense(n),
    tf.keras.layers.Dense(2)
])

#binary cross entropy loss using tensorflow
loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=Y_true, logits=Y_pred))

#mean squared error loss using tensorflow
loss = tf.reduce_mean(tf.square(tf.subtract(Y_true, Y_pred)))
loss = tf.keras.losses.MSE(Y_true, Y_pred)

#gradient descent optimizer using tensorflow
weights = tf.Variable(tf.random.normal([]))
while True:
    with tf.GradientTape() as tape:
        loss = compute_loss(weights)
    gradient = tape.gradient(loss, [weights]) #compute gradient means backpropagation

    weights = weights - learning_rate * gradient[0]

#adaptive learning rate optimizers using tensorflow
tf.keras.optimizers.SGD()
tf.keras.optimizers.Adam()
tf.keras.optimizers.Adadelta()
tf.keras.optimizers.Adagrad()
tf.keras.optimizers.RMSprop()

#regularization using tensorflow
tf.keras.layers.Dropout(rate=0.5)

#pythorch implementation of a dense layer
import torch
class MyDenseLayer(nn.Module):
    def __init__(self, input_dim,output_dim):
        super(MyDenseLayer, self).__init__()
        #initialize weights and biases
        self.w = nn.Parameter(torch.randn(input_dim, output_dim,requires_grad=True))
        self.b = nn.Parameter(torch.randn(1,output_dim,requires_grad=True))
    def forward(self, inputs):
        z = torch.matmul(inputs, self.w) + self.b #forward propagation
        output = torch.sigmoid(z) #activation function
        return output
    
#simplified call to a dense layer using pytorch
import torch.nn as nn
layer = nn.Linear(in_features=m, out_features=2)

#multiple output perceptron using pytorch
import torch.nn as nn
model = nn.Sequential(
    nn.Linear(m, n),
    nn.ReLU(),
    nn.Linear(n, 2)
)

#binary cross entropy loss using pytorch
loss = torch.nn.functional.cross_entropy(Y_pred, Y_true)

#mean squared error loss using pytorch
loss = torch.nn.functional.mse_loss(Y_pred, Y_true)

#adaptive learning rate optimizers using pytorch
torch.optim.SGD()
torch.optim.Adam()
torch.optim.Adadelta()
torch.optim.Adagrad()
torch.optim.RMSprop()

#regularization using pytorch
torch.nn.Dropout(p=0.5)
#final algorithm for training a neural network
import tensorflow as tf
model = tf.keras.Sequential([
    tf.keras.layers.Dense(n),
    tf.keras.layers.Dense(2)
])
optimizer = tf.keras.optimizers.Adam()
while True: #loop forever
    #forward pass
    predictions = model(X)
    with tf.GradientTape() as tape:
        loss = compute_loss(y, prediction)
    #update the weights using the gradient 
    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))
