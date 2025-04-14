from app.application.interfaces.i_ai_submission_service import IAISubmissionService
from app.infrastructure.graph.graph_builder import GraphBuilder
from langchain_core.messages import HumanMessage


class AISubmissionService(IAISubmissionService):
    """
    Service for handling AI submissions using a prebuilt graph.
    """

    def __init__(self, llm=None, retrieve_tool=None):
        """Initialize the service with a graph builder."""
        self.graph_builder = GraphBuilder(llm=llm, retrieve_tool=retrieve_tool)
        self.graph = self.graph_builder.build_graph()

    def process_message(self, input_message):
        """
        Process a message through the graph and stream responses.

        Args:
            input_message (str): User input message

        Returns:
            list: List of response messages
        """
        human_message = HumanMessage(content=input_message)

        responses = []
        for step in self.graph.stream(
            {"messages": [human_message]},
            stream_mode="values",
            config=self.graph_builder.config,
        ):
            step["messages"][-1].pretty_print()

            responses.append(step["messages"][-1])

        return responses
