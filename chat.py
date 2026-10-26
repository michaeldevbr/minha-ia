from langchain_ollama import ChatOllama
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage, ToolMessage
from langchain_core.tools import tool

from personalidade import PERSONALIDADE
from pathlib import Path
from datetime import datetime

PASTA_HISTORICO_CHATS = Path("historico_chats")
PASTA_HISTORICO_CHATS.mkdir(parents=True, exist_ok=True)

model = ChatOllama(
    model="llama3.2:3b",
    temperature=0.7,
    timeout=30,
    max_retries=6,
    num_ctx=8192,
)

historico = [
    SystemMessage(content=PERSONALIDADE),
]

@tool
def multiplicar(a: float, b: float) -> float:
    """Multiplica dois números."""
    return a * b


@tool
def somar(a: float, b: float) -> float:
    """Soma dois números."""
    return a + b

ferramentas = [
    multiplicar, 
    somar
]

registro_de_ferramentas = {
    ferramenta.name: ferramenta
    for ferramenta in ferramentas
}

modelo_com_ferramentas = model.bind_tools(ferramentas)

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

    if not mensagem:
        continue

    historico.append(HumanMessage(content=mensagem))
    resposta = modelo_com_ferramentas.invoke(historico)
    historico.append(resposta)

    quantidade_de_ciclos = 0
    limite_de_ciclos = 5

    while resposta.tool_calls and quantidade_de_ciclos < limite_de_ciclos:
        if resposta.tool_calls:
            for chamada_pendente in resposta.tool_calls:
                historico.append(
                    ToolMessage(
                        content="Execução interrompida: limite de ciclos atingido.",
                        tool_call_id=chamada_pendente["id"]
                    )
                )
        for chamada in resposta.tool_calls:
            nome_ferramenta = chamada["name"]
            argumentos_ferramenta = chamada["args"]
            id_ferramenta = chamada["id"]

            ferramenta = registro_de_ferramentas.get(nome_ferramenta)

            if ferramenta is None:
                resultado = (
                    f"Erro: a ferramenta '{nome_ferramenta}' "
                    "não está disponível ou não existe."
                )
            else:
                try:
                    resultado = ferramenta.invoke(argumentos_ferramenta)
                except Exception as erro:
                    resultado = (
                        f"Falha na execução da ferramenta "
                        f"'{nome_ferramenta}'. Motivo: {erro}"
                    )

            historico.append(
                ToolMessage(
                    content=str(resultado),
                    tool_call_id=id_ferramenta
                )
            )

        quantidade_de_ciclos += 1

        resposta = modelo_com_ferramentas.invoke(historico)
        historico.append(resposta)

    if resposta.tool_calls:
        print("Luna: Atingi o limite de chamadas de ferramentas.")
    else:
        print(f"Luna: {resposta.content}\n")
