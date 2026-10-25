from langchain_ollama import ChatOllama
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage

from personalidade import PERSONALIDADE
from pathlib import Path
from datetime import datetime

PASTA_HISTORICO_CHATS = Path("historico_chats")
PASTA_HISTORICO_CHATS.mkdir(parents=True, exist_ok=True)

model = ChatOllama(
    model="llama3.2:3b",
    temperature=0.7,
    timeout=30,
    max_retries=6
)

historico = [
    SystemMessage(content=PERSONALIDADE),
]

while True:
    mensagem = input("Você: ").strip()

    if(mensagem.lower() == "sair"):
        linhas = []

        for linha in historico:
            if isinstance(linha, SystemMessage):
                continue
            elif isinstance(linha, HumanMessage):
                linhas.append(f"Você: {linha.content}")
            elif isinstance(linha, AIMessage):
                linhas.append(f"Luna: {linha.content}")

        nome_arquivo = (
            f"historico_chat_{datetime.now():%Y-%m-%d_%H-%M-%S}.txt"
        )

        arquivo = PASTA_HISTORICO_CHATS / nome_arquivo
        arquivo.write_text("\n\n".join(linhas), encoding="utf-8")

        print(f"Chat salvo em {arquivo}")
        print("Luna: Bye bye!")
        break

    if(not mensagem):
        continue

    historico.append(HumanMessage(content=mensagem))
    resposta = model.invoke(historico)
    historico.append(AIMessage(content=resposta.content))

    print(f"Luna: {resposta.content}\n")
