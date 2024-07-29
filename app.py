import gradio as gr
import tensorflow as tf
import numpy as np

# model = tf.saved_model.load("/home/user/app/model")
model = tf.saved_model.load("C:\\Users\\caleb\\OneDrive\\Documents\\Coding Projects\\INFX_639\\mnist_checker\\model")
labels = ["zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine"]

def mnist_classifier(img):
    img_tensor = tf.convert_to_tensor(img['composite'], dtype=tf.float32)
    img_tensor = tf.expand_dims(img_tensor[:, :, -1], axis=-1) # Select the last channel
    img_tensor = tf.image.resize(img_tensor, [28, 28])  

    # Normalize and flatten
    img_tensor /= 255.0
    img_tensor = tf.reshape(img_tensor, (1, 784))

    prediction = model.signatures['serving_default'](img_tensor)
    prediction = tf.argmax(prediction['output'], axis=1).numpy()
    return str(prediction[0])

demo = gr.Interface(fn=mnist_classifier, inputs="sketchpad", outputs="text", title="MNIST Checker", description="Draw a number 0-9 to see if the model can classify it.")
demo.launch()