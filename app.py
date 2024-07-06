import gradio as gr
import tensorflow as tf

#model = tf.keras.models.load_model("mnist_checker/model/saved_model.pb")

def mnist_classifier(img):
    #img = tf.image.resize(img, [28, 28])
    #img = tf.cast(img, tf.float32)
    print(img['background'])
    print(img['composite'])
    return "returned: " + img['layers']

demo = gr.Interface(fn=mnist_classifier, inputs="sketchpad", outputs="text", title="MNIST Checker", description="Draw a number 0-9 to see if the model can classify it.")
demo.launch()