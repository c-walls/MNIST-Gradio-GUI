import gradio as gr
import tensorflow as tf

#model = tf.keras.models.load_model("mnist_checker/model/saved_model.pb")

def mnist_classifier(img):
    #img = tf.image.resize(img, [28, 28])
    #img = tf.cast(img, tf.float32)
    return img

demo = gr.Interface(fn=mnist_classifier, inputs="sketchpad", outputs="image")
demo.launch()