import gradio as gr
from transformers import pipeline

def convert(Sentence):
    model = pipeline('sentiment-analysis')
    result = model(Sentence)
    return result[0]['label'], round(result[0]['score']*100, 2)

with gr.Blocks() as bls:
    gr.Markdown("Here is a Sentiment Analysis app")
    with gr.Row():
        inp = gr.Textbox(label="Type your sentence here and click Run", placeholder='Example: the food delivered was stale')
        out1 = gr.Textbox(label="The sentiment is:")
        out2 = gr.Number(label='Sentiment confidence % is:')
    btn = gr.Button("Run")
    btn.click(fn=convert, inputs=inp, outputs=[out1, out2])

bls.launch()