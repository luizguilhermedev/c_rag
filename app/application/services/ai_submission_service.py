from langgraph.checkpoint.memory import MemorySaver
from app.graph.graph_builder import GraphBuilder
from app.infrastructure.initialize_llm import initialize_llm
from app.application.tools.retrieve_tool import retriever_tool
from langchain.schema import AIMessage


class AISubmissionService:
    """
    Service for handling AI submissions using a prebuilt graph.
    """

    def __init__(self):
        # Initialize dependencies
        self.llm = initialize_llm()
        self.retrieve_tool = retriever_tool

        self.graph_builder = GraphBuilder(self.llm, self.retrieve_tool)
        self.graph = self.graph_builder.build_graph().compile(
            checkpointer=MemorySaver()
        )

    def process_submission(self, input_message: str, config: dict) -> AIMessage:
        """
        Process an AI submission using the graph.

        Args:
            input_message (str): The input message from the user.
            config (dict): Configuration for the graph execution.

        Returns:
            AIMessage: The last AIMessage from the processed messages.
        """
        responses = []
        for step in self.graph.stream(
            {"messages": [{"role": "user", "content": input_message}]},
            stream_mode="values",
            config=config,
        ):
            responses.append(step["messages"][-1])

        # Filtrar e retornar o último AIMessage
        ai_messages = [msg for msg in responses if isinstance(msg, AIMessage)]
        return ai_messages[-1] if ai_messages else None


# Exemplo de uso
# submission_service = AISubmissionService()
# config = {"configurable": {"thread_id": "ab"}}
# response = submission_service.process_submission("What Darwin found in galapagos", config)
# if response:
#     print("Último AIMessage:", response.content)
# else:
#     print("Nenhum AIMessage encontrado.")
