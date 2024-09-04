from utils import bot
import gradio as gr

def AskQ(question,history):
    answer = bot(question=question)
    
    return answer

gr.ChatInterface(AskQ).launch()


# #define gradio interface and other parameters
# app =  gra.Interface(fn = AskQ, inputs="text", outputs="text")
# app.launch()



