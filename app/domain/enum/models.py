from enum import Enum

class ChatModels(str, Enum):

    GPT_4_o_mini = "gpt-4o-mini"
    GPT_4_o = "gpt-4o"


class ModelProviders(str, Enum):
    OPENAI = "openai"
    HUGGINGFACE = "huggingface"
    LOCAL = "local"