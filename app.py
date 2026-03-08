prompt = f"""
Responda usando apenas o conteúdo do PDF abaixo.

PDF:
{texto[:4000]}

Pergunta:
{pergunta}
"""

resposta = client.chat.completions.create(
    model="llama3-8b-8192",
    messages=[
        {"role": "user", "content": prompt}
    ],
    max_tokens=400,
    temperature=0.3
)

st.write(resposta.choices[0].message.content)
