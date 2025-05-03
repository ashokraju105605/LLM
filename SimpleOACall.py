import openai

def ask(question, context="", system=""):

    # Set default System Message
    if system == "":
        system = """You are an expert in Artificial Intelligence Research Papers. 
Use the following pieces of context to answer the users question. 
If you don't know the answer, just say that you don't know, don't try to make up an answer.
"""

    # Prepend context if used
    if context != "":
        question = "Use the following context to answer the users question:\n```\n" + context + "\n```\n\n" + question

    response = openai.ChatCompletion.create(
        engine="gpt-35-turbo",
        messages = [{"role":"system","content":system},{"role":"user","content":question}],
        temperature=0.0,
        max_tokens=500,
        top_p=0.95,
        frequency_penalty=0,
        presence_penalty=0,
        stop=None)
        
    return response['choices'][0]['message']['content']