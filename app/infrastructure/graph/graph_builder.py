from uuid import uuid4
from langchain_core.messages import SystemMessage
from langgraph.graph import END, MessagesState, StateGraph
from langchain.chat_models import init_chat_model
from langgraph.prebuilt import ToolNode, tools_condition
from app.application.tools.retrieve_tool import retriever_tool as retrieve
from langgraph.checkpoint.memory import MemorySaver


class GraphBuilder:
    """
    Builds the state graph for AI submissions.
    """

    def __init__(self, llm=None, retrieve_tool=None, thread_id: uuid4 = None):
        if llm is None:
            self.llm = init_chat_model("gpt-4o-mini", model_provider="openai")
        else:
            self.llm = llm

        self.retrieve_tool = retrieve_tool if retrieve_tool else retrieve
        self.graph_builder = StateGraph(MessagesState)
        self.memory = MemorySaver()

        if thread_id is None:
            thread_id = str(uuid4())
            
        self.config = {"configurable": {"thread_id": thread_id}}

    def query_or_respond(self, state: MessagesState):
        """Generate tool call for retrieval or respond."""
        llm_with_tools = self.llm.bind_tools([self.retrieve_tool])
        response = llm_with_tools.invoke(state["messages"])
        return {"messages": [response]}

    def generate(self, state: MessagesState):
        """Generate answer."""
        recent_tool_messages = []
        for message in reversed(state["messages"]):
            if message.type == "tool":
                recent_tool_messages.append(message)
            else:
                break
        tool_messages = recent_tool_messages[::-1]

        docs_content = "\n\n".join(doc.content for doc in tool_messages)
        system_message_content = (
            "You are an assistant for question-answering tasks related to the well know book 'The origin of the species'."
            "Use the following pieces of retrieved context to answer "
            "the question. If you don't know the answer, say that you "
            "don't know."
            "Keep the politeness and the tone of the conversation."
            "\n\n"
            f"{docs_content}"
        )
        conversation_messages = [
            message
            for message in state["messages"]
            if message.type in ("human", "system")
            or (message.type == "ai" and not message.tool_calls)
        ]
        prompt = [SystemMessage(content=system_message_content)] + conversation_messages

        response = self.llm.invoke(prompt)
        return {"messages": [response]}

    def build_graph(self):
        """Build and compile the graph."""
        tools = ToolNode([self.retrieve_tool])

        self.graph_builder.add_node(self.query_or_respond)
        self.graph_builder.add_node(tools)
        self.graph_builder.add_node(self.generate)

        self.graph_builder.set_entry_point("query_or_respond")
        self.graph_builder.add_conditional_edges(
            "query_or_respond",
            tools_condition,
            {END: END, "tools": "tools"},
        )
        self.graph_builder.add_edge("tools", "generate")
        self.graph_builder.add_edge("generate", END)

        graph = self.graph_builder.compile(checkpointer=self.memory)

        return graph
