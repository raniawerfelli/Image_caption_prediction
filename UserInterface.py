import gradio as gr
from predict_caption import predict_step
#def de l'interf gradio
with gr.Blocks() as demo:
     #Ajout des composants
     i_image = gr.Image(type='pil', label='Image')
     label = gr.Text(label='Generated Caption')
     french_label = gr.Textbox(label='French Translation')
     arabic_label = gr.Textbox(label='Arabic Translation')
    
     i_image.upload(
        predict_step,
        [i_image],
        [label, french_label, arabic_label]
     )

if __name__ == '__main__':
    demo.launch()
