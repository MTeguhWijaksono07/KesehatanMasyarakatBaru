def build_prompt(context, question):

    prompt = f'''
    Anda adalah asisten kesehatan masyarakat.

    Konteks:
    {context}

    Pertanyaan:
    {question}

    Jawaban:
    '''

    return prompt
