import streamlit as st
import transformers
import model
from model import load_model
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from wordpress_xmlrpc import Client, WordPressPost
from wordpress_xmlrpc.methods.posts import GetPosts, NewPost
from wordpress_xmlrpc.methods.users import GetUserInfo

def infer(input_ids, max_length, temperature):

    output_sequences = model.generate(
        input_ids=input_ids,
        max_length=800,
        temperature=temperature,
        do_sample=True,
    )
    return output_sequences

#Prompts
st.title("Generate code with GPT codegen 🦄")
st.subheader("This machine learning model is trained on 16 billion parameters and it generates code in every language for your development projects to get inspired with new ideas or get insight or when you are stuck with google, this can help you further!")

generate = st.button("GENERATE CODE")
text_target = st.text_area(label = "Enter your instruction and leave the answer open for the generated code, you can use algorithms to generate code of a maximum length of 800 or you can put your code inside the AI and let the AI generate new code", value ="""Instruction: Generate python pytorch audio code for a GAN that generate MP3
Answer:""", height = 300)
temperature = st.slider("Temperature", value = 0.9, min_value = 0.0, max_value=1.0, step=0.1)
max_length = 800

#Load model and generate
if generate:
    with st.spinner("AI is at work......"):
        model, tokenizer, device = load_model()
        input_ids = tokenizer(text=text_target, return_tensors="pt").input_ids.to(device)
        output_sequences = infer(input_ids, max_length, temperature)
        generated_ids = model.generate(output_sequences)
        generated_text = tokenizer.decode(generated_ids[0])
        wp = Client('https://bliknotes.com/xmlrpc.php', 'Blik', '!Uk6fPu!5Qv*m24cKQvwz4YS')
        post = WordPressPost()
        post.title = text_target
        post.content = """[code language="actionscript3"]""" + generated_text + """[/code]"""
        post.post_status = 'publish'
        wp.call(NewPost(post))
    st.success("AI Succesfully generated code")
    print(generated_text) 

    st.text(generated_text)