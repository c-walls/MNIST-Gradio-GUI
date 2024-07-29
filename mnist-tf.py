import tensorflow as tf
import numpy as np
import os

# Load the MNIST dataset
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

# Normalize and reshape the data
x_train, x_test = x_train.astype(np.float32) / 255.0, x_test.astype(np.float32) / 255.0
x_train = x_train.reshape(-1, 784)
x_test = x_test.reshape(-1, 784)

# Convert labels to one-hot encoding
y_train = tf.one_hot(y_train, depth=10)
y_test = tf.one_hot(y_test, depth=10)

# Create TensorFlow datasets for better performance
train_dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train)).shuffle(60000).batch(100)
test_dataset = tf.data.Dataset.from_tensor_slices((x_test, y_test)).batch(100)

# Network parameters
input_layer_size = 784
hidden_layer_one = 256
hidden_layer_two = 256
number_classes = 10

weights = {
    'w1': tf.Variable(tf.random.normal([input_layer_size, hidden_layer_one], dtype=tf.float32)),
    'w2': tf.Variable(tf.random.normal([hidden_layer_one, hidden_layer_two], dtype=tf.float32)),
    'w_out': tf.Variable(tf.random.normal([hidden_layer_two, number_classes], dtype=tf.float32))
}

biases = {
    'b1': tf.Variable(tf.random.normal([hidden_layer_one], dtype=tf.float32)),
    'b2': tf.Variable(tf.random.normal([hidden_layer_two], dtype=tf.float32)),
    'b_out': tf.Variable(tf.random.normal([number_classes], dtype=tf.float32))
}

# Network architecture
def feedforward_network(x):
    layer_1 = tf.nn.relu(tf.add(tf.matmul(x, weights['w1']), biases['b1']))
    layer_2 = tf.nn.relu(tf.add(tf.matmul(layer_1, weights['w2']), biases['b2']))
    output_layer = tf.matmul(layer_2, weights['w_out']) + biases['b_out']
    return output_layer

# Training hyperparameters
epochs = 45
learning_rate = 0.001
job_dir = 'mnist_model'

# Training loop
for epoch in range(epochs):
    for step, (batch_x, batch_y) in enumerate(train_dataset):
        with tf.GradientTape() as tape:
            logits = feedforward_network(batch_x)
            loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=batch_y))
        
        gradients = tape.gradient(loss, list(weights.values()) + list(biases.values()))
        optimizer = tf.optimizers.Adam(learning_rate=learning_rate)
        optimizer.apply_gradients(zip(gradients, list(weights.values()) + list(biases.values())))
    
    # Print loss every epoch
    print(f"Epoch {epoch+1}, Loss: {loss.numpy()}")

# Evaluation
def evaluate(dataset):
    correct_predictions = 0
    total_predictions = 0
    for batch_x, batch_y in dataset:
        logits = feedforward_network(batch_x)
        correct_predictions += tf.reduce_sum(tf.cast(tf.equal(tf.argmax(logits, 1), tf.argmax(batch_y, 1)), tf.int32)).numpy()
        total_predictions += batch_x.shape[0]
    return correct_predictions / total_predictions

accuracy = evaluate(test_dataset)
print(f"Test accuracy: {accuracy}")

@tf.function(input_signature=[tf.TensorSpec(shape=[None, 784], dtype=tf.float32)])
def serve_model(x):
    return {'output': feedforward_network(x)}

# Save the model
class MyModel(tf.Module):
    def __init__(self, weights, biases):
        super(MyModel, self).__init__()
        self.weights = weights
        self.biases = biases
        self.serve_model = serve_model

model = MyModel(weights, biases)
save_path = os.path.join(job_dir, 'model')
tf.saved_model.save(model, save_path)