import gradio as gr
import tensorflow as tf
import numpy as np

model = tf.saved_model.load("/home/user/app/model")
# Print available signatures to find the correct one
print("Available signatures:", list(model.signatures.keys()))

def mnist_classifier(img):
    img_tensor = tf.convert_to_tensor(img['composite'], dtype=tf.float32)
    img_tensor = tf.image.resize(img_tensor, [28, 28])
    
    # Normalize and expand to add the batch dimension
    img_tensor /= 255.0
    img_tensor = tf.expand_dims(img_tensor, 0)

    #prediction = model.signatures['serving_default'](img_tensor)
    #print(model.signatures['serving_default'].structured_outputs)
    #return str(tf.argmax(prediction['output_0'], axis=1).numpy()[0])

demo = gr.Interface(fn=mnist_classifier, inputs="sketchpad", outputs="text", title="MNIST Checker", description="Draw a number 0-9 to see if the model can classify it.")
demo.launch()