from langchain.chat_models import init_chat_model


def initialize_llm():
    """Initialize the LLM."""
    return init_chat_model("gpt-4o-mini", model_provider="openai")
