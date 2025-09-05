import os
import base64
import tempfile

from openai import AzureOpenAI
from dotenv import load_dotenv
from fastapi import FastAPI, UploadFile, Form
from fastapi.responses import JSONResponse
from pathlib import Path
from fastapi.middleware.cors import CORSMiddleware

env_path = Path(__file__).resolve(strict=True).parent / '.env'
load_dotenv(dotenv_path=env_path)

# Carregando variáveis do arquivo de ambiente
AZURE_OPENAI_API_KEY = os.getenv("AZURE_OPENAI_API_KEY")
AZURE_OPENAI_ENDPOINT = os.getenv*("AZURE_OPENAI_ENDPOINT")
AZURE_OPENAI_API_VERSION = os.getenv("AZURE_OPENAI_API_VERSION")
AZURE_OPENAI_DEPLOYMENT_NAME = os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME")


app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

client = AzureOpenAI(
    api_key=AZURE_OPENAI_API_KEY,
    azure_endpoint=AZURE_OPENAI_ENDPOINT,
    api_version=AZURE_OPENAI_API_VERSION,
    azure_deployment=AZURE_OPENAI_DEPLOYMENT_NAME)

def criar_prompt_modelo_ameacas(tipo_aplicacao: str, 
                                autenticacao: str, 
                                acesso_internet: bool, 
                                dados_sensiveis: bool, 
                                descricao_aplicacao:str):

    prompt = f"""Aja como um especialista em cibersegurança com mais e 20 anos de experiência 
    utilizando a metodologia de modelagem de ameaças STRIDE para produzir modelos de ameaças abrangentes
    para uma ampla gama de aplicações. Sua tarefa é analisas o resumo do dcódigo,
    o conteúdo do README e a descrição da aplicação fornecidos para produzir uma lista de ameaçãs específicas para essa aplicação.
    
    Presta atenção na descrição da aplicação e nos detalhes técnicos fornecidos.

    Para cada uma das categorias STRIDE (Falsificação de Identidade - Spoofing, 
    Violação de Integridade - Tampering, 
    Repúdio - Repudiation, 
    Divulgação de Informação - Information Disclosure,
    Negação de Serviço -Denial of Service,
    Elevação de Privilégios - Elevation of Privilege), liste múltiplas (3 ou 4) ameaças crediveis se aplicável. 
    Cada cenário de ameaça deve apresentar uma situação plausível em que a amaça poderia ocorrer no contexto da aplicação.

    A lista de ameaças deve ser apresentada em formato de tabela com as seguinte colunas:
    Ao fornecer o modelo de ameaças, utilize uma resposta no formato JSON, com as chaves "threat_model" e "improvement_suggestions.
    Em "threat_model", inclua array com objetos com as chaves "Threat Type" (Tipo de Ameaça), "Scenario" (Cenário), e
    "    "Potential Impact" (Impacto Potencial).
    Impact" (Impacto Potencial).

    Ao fornecer o modelo de ameaças, utilize uma resposta em formato JSON com as chaves "threat_model" e "improvement_suggestions".
    Em "threat_model", inclua array com objetos com as chaves "Threat Type" (Tipo de Ameaça), "Scenario" (Cenário), e
    "Potential Impact" (Impacto Potencial).
    
    Em "improvement_suggestions", forneça um array de sugestões práticas para mitigar as ameaças identificadas.

    TIPO DE APLICAÇÃO: {tipo_aplicacao}
    MÉTODOS DE AUTENTICAÇÃO: {autenticacao}
    APLICAÇÃO POSSUI ACESSO À INTERNET: {acesso_internet}
    APLICAÇÃO MANIPULA DADOS SENSÍVEIS: {dados_sensiveis}
    RESUMO DO CÓDIGO, CONTEÚDO DO README E A DESCRIÇÃO DA APLICAÇÃO: {descricao_aplicacao}

    Exemplode de Formatação JSON:
    {{
        "threat_model": [
            {{
                "Threat Type": "Spoofing",
                "Scenario": "An attacker impersonates a legitimate user to gain unauthorized access.",
                "Potential Impact": "Unauthorized access to sensitive data."
            }},
            {{
                "Threat Type": "Tampering",
                "Scenario": "An attacker modifies data in transit to alter its meaning.",
                "Potential Impact": "Data integrity is compromised, leading to incorrect decisions."
            }}
        ],
        "improvement_suggestions": [
            "Implement multi-factor authentication to enhance security.",
            "Use encryption for data in transit and at rest."
        ]
    }}
    """

    return prompt
