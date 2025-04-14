# Arquitetura

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

2. **Suporte a JSON**: Utilização de um Agent para processar o documento de texto(txt) em um JSON estruturado:
```
    {
        {
            "book_content": "The main book content...",
            "summary_content": "Any summaries and chapter overviews..."
        }
    }
```


# API

O modelo conversacional é disponibilizado via API FastAPI.
Também temos a API de maneira conteinerizada via Docker.



# Execução do Projeto

Este projeto utiliza uv para gerenciamento de dependências.

Criar ambiente virtual:

`uv venv`

Ativando o ambiente virtual:

`source .venv/bin/activate`

Gerando o requirements:

`make reqs`

Instalando as dependencias:

`make sync`

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
