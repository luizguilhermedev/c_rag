![Captura de tela 2025-04-14 093007](https://github.com/user-attachments/assets/67d21989-9368-4bbb-9bc9-d3f3155b5d7d)# Arquitetura

O projeto utiliza a arquitetura RAG - Retrieval Augmented Generation - que aprimora a geração de respostas de IA através da recuperação contextual de informações relevantes. O sistema processa documentos de entrada, extrai conhecimento estruturado e utiliza esse conhecimento para fornecer respostas mais precisas e fundamentadas.

# Design Pattern

O projeto está organizado seguindo o Design Pattern baseado em Clean Code. Possui camadas isoladas de domínio, infrastructure, aplicação e apresentação.

## Estrutura do Projeto

```
.
├── app/
│   ├── main.py
│   ├── settings.py
│   ├── application/
│   │   ├── interfaces/
│   │   │   └── i_ingestion_service.py
│   │   ├── tools/
│   │   │   └── retrieve_tool.py
│   │   ├── services/
│   │   │   ├── ingestion_service.py
│   │   │   └── ai_submission_service.py
│   ├── domain/
│   │   ├── entities/
│   │   │   ├── chunk.py
│   │   │   ├── document.py
│   │   │   └── embedding.py
│   │   └── interfaces/
│   │       ├── i_document_processor.py
│   │       ├── i_embedding_provider.py
│   │       └── i_vector_store.py
│   ├── infrastructure/
│   │   ├── graph/
│   │   │   └── graph_builder.py
│   │   ├── processors/
│   │   │   ├── preprocessor_document_agent.py
│   │   │   └── text_document_processor.py
│   │   ├── vector_store/
│   │   │   └── chroma_vector_store.py
│   │   └── initialize_llm.py
│   └── presentation/
│       ├── endpoints/
│       │   └── ai_submission_endpoint.py
│       └── models/
│           ├── submission_request.py
│           └── submission_response.py
├── data/
│   ├── origin_of_species.txt
│   └── chroma_db
├── output/
│   └── results.json
├── processed_documents/
├── requirements/
│   └── prd.txt
├── pyproject.toml
└── README.md
```



# Ingestão de dados para arquitetura RAG

O projeto possui duas abordagens para a ingestão de dados no sistema RAG proposto.

1. **Processamento de documentos em formato original**: Utilização do texto em seu formato original(txt) para geração de chunks e embeddings.

documento.txt
   │
   ▼
Limpeza do texto
   │
   ▼
Chunking (SemanticChunker - LangChain)
   │
   ▼
Geração de Embeddings (OpenAI)
   │
   ▼
Armazenamento no Banco Vetorial (ChromaDB)

![Captura de tela 2025-04-14 091631](https://github.com/user-attachments/assets/547ed71a-a9a4-405f-bfdd-362620306127)


2. **Suporte a JSON**: Utilização de um Agent para processar o documento de texto(txt) em um JSON estruturado:

![Captura de tela 2025-04-14 091543](https://github.com/user-attachments/assets/d3a1e33c-97bb-49aa-af84-8b5f880dae38)

```
    {
        {
            "book_content": "The main book content...",
            "summary_content": "Any summaries and chapter overviews..."
        }
    }
```

# Estratégia RAG

O sistema segue uma arquitetura baseada em agentes com fluxo controlado por grafo (LangGraph), composta por duas etapas principais:

1. Retrieval (Ingestão e Ferramenta de Busca)

Normalização e pré-processamento do conteúdo textual.

Segmentação semântica com SemanticChunker (via LangChain).

Geração de embeddings utilizando modelos da OpenAI.

Armazenamento vetorial no ChromaDB para posterior recuperação.

O ToolNode encapsula o mecanismo de busca, ativado dinamicamente via LangGraph.

2. Generation (Fluxo com LLM e LangGraph)
O serviço AISubmissionService inicializa o LLM, define o retriever_tool e constrói o grafo com GraphBuilder.

Ao receber um input_message, o método process_submission itera sobre o grafo compilado (self.graph).

Dentro do fluxo do LangGraph:

O nó query_or_respond decide se o LLM deve responder diretamente ou invocar ferramentas.

Caso ferramentas sejam necessárias (tools_condition), o fluxo segue para o ToolNode que executa a busca com base no contexto.

A resposta é gerada

![Captura de tela 2025-04-14 093007](https://github.com/user-attachments/assets/f03adf4b-b436-4a54-bc8a-e7b0616e8e83)


# API

O modelo conversacional é disponibilizado via API FastAPI.
Também temos a API de maneira conteinerizada via Docker.



# Execução do Projeto

Este projeto utiliza uv para gerenciamento de dependências.

* Criar ambiente virtual:

`uv venv`

Ativando o ambiente virtual:

`source .venv/bin/activate`

* Gerando o requirements:

`make reqs`

* Instalando as dependencias:

`make sync`

* Crie um arquivo .env na raiz do projeto:

`touch .env`

Adicione as seguintes variáveis de ambiente:

OPENAI_API_KEY=sua_chave_api_openai

## "Rodando" a API via DOCKER

Certificar que está com o docker instalado:

`docker --version`

Utilizar docker compose para subir a api:

`docker compose up`


Acessar o Swagger em:

http://localhost:8000/docs

Request Body:
```
{
  "input_message": "string",
  "config": {
    "additionalProp1": {} #Preencher com "additionalProp1": {"thread_id": "id"}
  }
}
```

O thread_id é utilizado para manter o histórico de conversa com o chatbot.

## Testar a API localmente - DOCKER
```
curl -X POST http://127.0.0.1:8000/api/v1/darwin-chat-bot/ai-submission \
  -H "Content-Type: application/json" \
  -d '{
    "input_message": "Qual é a importância da variação nas espécies domesticadas?",
    "config": {
      "configurable": {"thread_id":"sua_thread_id"}
    }
}'
```

## Testar API localmente com debugger

Executar o debugger com o arquivo launcher.json

# Requisição de exemplo com cURL
```
curl -X POST http://127.0.0.1:8000/api/v1/darwin-chat-bot/ai-submission \
  -H "Content-Type: application/json" \
  -d '{
    "input_message": "Qual é a importância da variação nas espécies domesticadas?",
    "config": {
      "configurable": {"thread_id":"sua_thread_id"}
    }
}'
```

# Logs e Troubleshooting

Caso enfrente algum problema ao subir a aplicação:

- Verifique se as portas 8000 e 3000 estão livres.
- Consulte os logs do container:

```bash
docker logs <nome-do-container>
```

# Tecnologias e Frameworks Utilizados
O projeto é construído utilizando as seguintes tecnologias e frameworks:

## Core
- Python 3.10+: Linguagem de programação principal
- FastAPI: Framework web para construção da API REST
- LangChain: Biblioteca para operações com LLMs e processamento de documentos
- LangGraph: Controle de fluxo conversacional com memória
- OpenAI API: Geração de embeddings e interação com modelos LLM

## Armazenamento e Processamento
- ChromaDB: Banco de dados vetorial para armazenamento e recuperação de embeddings
- SemanticChunker: Ferramenta de chunking de textos baseada em significado
## DevOps
- Docker: Conteinerização da aplicação
- Docker Compose: Orquestração de contêineres
- uv: Gerenciador de dependências Python
- GitHub Actions: CI para verificação de lint

## Bibliotecas Auxiliares
- Pydantic: Validação de dados e configurações
- Uvicorn: Servidor ASGI para executar a aplicação FastAPI

# Melhorias

Por se tratar de um case técnico e quase uma POC, existem diversas melhorias que podem ser feitas ao longo do tempo com o produto.

- Utilização de arquitetura **Self Reflective Agent** para garantir que o Agent garanta que o conteúdo recuperado está dentro dos padrões esperados.
- Criação e implementação de Tools personalizadas para que os Agents possam consultar APIs externas e motores de busca externos, melhorando ainda mais a qualidade do conteúdo recuperado.
    - Poderiamos ter uma API com n estratégias de busca no banco vetorial e o Agent utilizaria essa API para recuperar conteúdo de maneira muito mais eficaz.
    - Consulta a APIs que disponibilizam livros - O Agent teria acesso ao material na íntegra, evitando perdas com chunking.
- Implementação de wrapper próprio para garantir maior segurança e controle dos dados - Embora o projeto já utilize padrão de arquitetura limpa(que isola as diversas partes do projeto), a criação de um wrapper próprio traria ainda mais flexibilidade, modularidade e segurança.
- Utilização de Banco Vetorial mais robusto como QDrant ou PineCone, que trariam maiores possibilidades e estratégias de busca vetorial e semântica.
- Criação de Retriever próprio ou de retrievers com maior precisão.
- CI/CD - Hoje o projeto conta com CI utilizando github actions - para verificação de lint. Com o projeto em produção, seria interessante implementar o CI para também criar imagens docker em serviços como ECR ou GCK. Também podemos criar o CD para deploy em um POD(exemplo ArgoCD) utilizando Kubernets para orquestração dos jobs. Assim teriamos o flow completo de CI/CD.
