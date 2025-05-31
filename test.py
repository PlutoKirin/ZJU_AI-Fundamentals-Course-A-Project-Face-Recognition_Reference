import gradio as gr

def cam_test():
    return gr.Image(source="webcam", streaming=True)

gr.Interface(fn=cam_test, inputs=None, outputs="image").launch()