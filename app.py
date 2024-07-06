import gradio as gr

def greet(name):
    return "Hi " + name + "!!"

demo = gr.Interface(fn=greet, inputs="sketchpad", outputs="label")
demo.launch()