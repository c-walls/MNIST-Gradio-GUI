import gradio as gr
import tensorflow as tf

model = tf.saved_model.load("/home/user/app/model")

def mnist_classifier(img):
    # Convert the image input to a tensor
    img_tensor = tf.convert_to_tensor(img['composite'], dtype=tf.float32)
    img_tensor = tf.image.resize(img_tensor, [28, 28])
    
    # Normalize and expand to add the batch dimension
    img_tensor /= 255.0
    img_tensor = tf.expand_dims(img_tensor, 0)

    #prediction = model.predict(img_tensor)
    return "Hi"

demo = gr.Interface(fn=mnist_classifier, inputs="sketchpad", outputs="text", title="MNIST Checker", description="Draw a number 0-9 to see if the model can classify it.")
demo.launch(share=True)