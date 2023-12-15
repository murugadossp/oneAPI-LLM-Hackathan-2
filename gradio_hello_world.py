import gradio as gr

# Function to be called by the interface
def greet(name):
    return f"Hello {name}!"

# Creating a Gradio interface
interface = gr.Interface(
    fn=greet,                           # function to call
    inputs=gr.components.Textbox(),     # input component
    outputs=gr.components.Textbox(),    # output component
)

# Launch the interface
interface.launch(server_port=7901)
