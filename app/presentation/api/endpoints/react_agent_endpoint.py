from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import List, Optional, Dict, Any

from app.domain.prompts.react_agent_service_prompt import react_agent_prompt
from app.application.services.react_agent_service import print_stream, agent
from app.logs import get_logger

# Create models for request and response
class ReactAgentRequest(BaseModel):
    input_message: str
    thread_id: Optional[str] = None

class RetrievedDocument(BaseModel):
    content: str
    metadata: Dict[str, Any] = {}

class ReactAgentResponse(BaseModel):
    content: str
    retrieved_docs: Optional[List[RetrievedDocument]] = None
    thread_id: Optional[str] = None

# Setup logging
logger = get_logger(__name__)


# Create router
router = APIRouter(prefix="/react-agent", tags=["ReAct Agent"])

@router.post("/query", response_model=ReactAgentResponse)
async def process_query(request: ReactAgentRequest):
    """
    Endpoint to process queries using the ReAct agent.
    """
    logger.info(
        f"API: Receiving request to process ReAct agent query: {request.input_message[:50]}..."
    )

    try:
        # Setup configuration for the agent
        config = {}
        if request.thread_id:
            config["configurable"] = {"thread_id": request.thread_id}
        
        # Prepare inputs for the agent
        inputs = {"messages": [{"role": "user", "content": request.input_message}]}
        
        # Invoke the agent
        result = print_stream(
            inputs=inputs,
            config=config,
            graph=agent,
        )
        
        # Process the response
        content = ""
        retrieved_docs = []
        
        # Extract the final message from the result
        final_message = result["messages"][-1]
        
        if hasattr(final_message, "content"):
            content = final_message.content
        else:
            content = str(final_message)
        
        # Extract any retrieved documents (from tool usage)
        for step in result.get("intermediate_steps", []):
            if hasattr(step, "tool") and step.tool in ["retriever_tool", "query_rewrite_tool"]:
                try:
                    # Attempt to extract document content and metadata
                    if hasattr(step, "tool_input") and hasattr(step, "tool_output"):
                        doc_content = step.tool_output
                        doc_metadata = getattr(step, "metadata", {})
                        retrieved_docs.append(
                            RetrievedDocument(content=doc_content, metadata=doc_metadata)
                        )
                except Exception as e:
                    logger.error(f"Error parsing tool output: {e}")
        
        # Get thread_id from config for response
        thread_id = config.get("configurable", {}).get("thread_id")
        
        return ReactAgentResponse(
            content=content,
            retrieved_docs=retrieved_docs if retrieved_docs else None,
            thread_id=thread_id,
        )

    except Exception as e:
        logger.error(f"Error processing ReAct agent query: {e}", exc_info=True)
        raise HTTPException(
            status_code=500, detail=f"Error processing ReAct agent query: {str(e)}"
        )