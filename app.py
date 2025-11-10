import gradio as gr
# from tokenizers import Tokenizer
from bengali_bpe_tokenizer import BengaliBPETokenizer
import os

# Load tokenizer saved by our custom implementation
# Ensure the file is uploaded alongside this app.py in the Space
TOKENIZER_PATH = "bengali_bpe_tokenizer.json"

tokenizer = BengaliBPETokenizer.load(TOKENIZER_PATH)

def tokenize_text(text):
    ids = tokenizer.encode(text)
    tokens = [tokenizer.inverse_vocab.get(i, '<UNK>') for i in ids]
    token_count = len(tokens)

    if len(text.strip()) == 0:
        comp_ratio = 0
    else:
        comp_ratio = len(text) / token_count if token_count > 0 else 0

    colored = ""
    colors = ["#e57373", "#81c784", "#64b5f6", "#ba68c8", "#ffb74d",
              "#4db6ac", "#9575cd", "#f06292", "#7986cb", "#aed581"]

    for i, t in enumerate(tokens):
        c = colors[i % len(colors)]
        colored += (
            f"<span style='background-color:{c}; padding:3px; margin:2px; "
            f"border-radius:4px;'>{t}</span>"
        )

    token_list = "\n".join([f"{t} ({id})" for t, id in zip(tokens, ids)])

    return token_count, round(comp_ratio, 4), colored, token_list


with gr.Blocks() as demo:
    gr.Markdown("<h1>Bengali BPE Tokenizer</h1>")

    with gr.Row():
        with gr.Column():
            text_input = gr.Textbox(
                label="Input Text",
                placeholder="Type Bengali text here...",
                lines=6
            )

        with gr.Column():
            token_count = gr.Number(label="Token Count")
            comp_ratio = gr.Number(label="Compression Ratio")

    tokens_html = gr.HTML(label="")
    token_list_output = gr.Textbox(label="Token List", lines=20)

    text_input.change(
        fn=tokenize_text,
        inputs=text_input,
        outputs=[token_count, comp_ratio, tokens_html, token_list_output]
    )

if __name__ == "__main__":
    demo.launch()
