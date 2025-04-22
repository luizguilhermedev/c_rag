from langchain.chat_models import init_chat_model
from app.domain.enum.models import (ChatModels, ModelProviders)

def initialize_llm():
    """Initialize the LLM."""
    return init_chat_model(model=ChatModels.GPT_4_o_mini, model_provider=ModelProviders.OPENAI)
