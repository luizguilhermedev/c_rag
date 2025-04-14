from typing import List, Optional, Dict, Tuple
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.messages import HumanMessage
from langgraph.graph import StateGraph, END
import json
import os
from pathlib import Path
from datetime import datetime
from pydantic import BaseModel

from app.domain.entities.chunk import Chunk
from app.settings import settings


class DocumentProcessorState(BaseModel):
    document_text: str
    document_name: str = "document"
    batches: List[str] = []
    current_batch_idx: int = 0
    book_content: str = ""
    summary_content: str = ""
    batch_results: List[Dict[str, str]] = []
    output_dir: str = "processed_documents"
    error: Optional[str] = None


def split_into_batches(state: DocumentProcessorState) -> DocumentProcessorState:
    """Split the document into manageable batches."""
    batch_size = 120000
    overlap = 1000
    text = state.document_text

    os.makedirs(state.output_dir, exist_ok=True)

    if len(text) <= batch_size:
        state.batches = [text]
        return state

    batches = []
    start = 0

    while start < len(text):
        if start + batch_size >= len(text):
            batches.append(text[start:])
            break

        end = start + batch_size
        paragraph_break = text.rfind("\n\n", start, end)
        if paragraph_break != -1 and paragraph_break > start + batch_size // 2:
            end = paragraph_break
        else:
            sentence_break = text.rfind(". ", start, end)
            if sentence_break != -1 and sentence_break > start + batch_size // 2:
                end = sentence_break + 1

        batches.append(text[start:end])
        start = end - overlap

    state.batches = batches
    return state


def process_batch(state: DocumentProcessorState) -> DocumentProcessorState:
    """Process the current batch using LLM."""
    if state.current_batch_idx >= len(state.batches):
        return state

    model = ChatOpenAI(
        model_name=settings.CHAT_MODEL,
        temperature=0,
        openai_api_key=settings.OPENAI_API_KEY,
    )

    batch_dir = os.path.join(state.output_dir, "batches", state.document_name)
    os.makedirs(batch_dir, exist_ok=True)

    prompt_template = PromptTemplate(
        input_variables=["document_text", "batch_number", "total_batches"],
        template="""
        You are preprocessing a document, in this case a book.
        This is batch {batch_number} of {total_batches}.

        Split the content into two categories:
        1. book_content: The main content of the book
        2. summary_content: Chapter summaries and overviews

        Return a JSON object, WITHOUT THE PATTERNS ```json``` and scaping characters, following this format:
        {{
            "book_content": "The main book content...",
            "summary_content": "Any summaries and chapter overviews..."
        }}

        DO NOT invent or alter content.

        Document text:
        {document_text}
        """,
    )

    batch = state.batches[state.current_batch_idx]
    formatted_prompt = prompt_template.format(
        document_text=batch,
        batch_number=state.current_batch_idx + 1,
        total_batches=len(state.batches),
    )

    try:
        response = model.invoke([HumanMessage(content=formatted_prompt)])

        batch_result = {}
        try:
            result = json.loads(response.content)
            batch_result = {
                "book_content": result.get("book_content", ""),
                "summary_content": result.get("summary_content", ""),
            }
        except json.JSONDecodeError:
            content = response.content
            start_idx = content.find("{")
            end_idx = content.rfind("}") + 1

            if start_idx >= 0 and end_idx > start_idx:
                try:
                    json_str = content[start_idx:end_idx]
                    result = json.loads(json_str)
                    batch_result = {
                        "book_content": result.get("book_content", ""),
                        "summary_content": result.get("summary_content", ""),
                    }
                except json.JSONDecodeError as e:
                    print(f"JSON parsing error within extract: {e}")
                    batch_result = {"book_content": batch, "summary_content": ""}
            else:
                batch_result = {"book_content": batch, "summary_content": ""}

        state.batch_results.append(batch_result)
        state.book_content += batch_result["book_content"]
        state.summary_content += batch_result["summary_content"]

        # Save intermediate batch result to JSON file
        batch_file = os.path.join(
            batch_dir, f"batch_{state.current_batch_idx + 1:03d}.json"
        )
        with open(batch_file, "w", encoding="utf-8") as f:
            json.dump(batch_result, f, ensure_ascii=False, indent=2)
            print(
                f"Saved batch {state.current_batch_idx + 1}/{len(state.batches)} to {batch_file}"
            )

    except Exception as e:
        state.error = f"Error processing batch {state.current_batch_idx}: {str(e)}"

    state.current_batch_idx += 1
    return state


def check_completion(state: DocumentProcessorState) -> str:
    """Check if all batches have been processed."""
    if state.error:
        return "handle_error"
    elif state.current_batch_idx >= len(state.batches):
        return "save_results"  # Go to save_results node when done
    else:
        return "continue"


def handle_error(state: DocumentProcessorState) -> DocumentProcessorState:
    """Handle any errors that occurred during processing."""
    print(f"Error during processing: {state.error}")
    return state


def save_results(state: DocumentProcessorState) -> DocumentProcessorState:
    """Save the final results as JSON for later chunking."""

    # Create the complete JSON that will be used for chunking
    final_json = {
        "book_content": state.book_content,
        "summary_content": state.summary_content,
        "metadata": {
            "document_name": state.document_name,
            "processed_date": datetime.now().isoformat(),
            "total_batches": len(state.batches),
            "error": state.error,
        },
    }

    # Save the final JSON to a file
    json_path = os.path.join(state.output_dir, f"{state.document_name}.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(final_json, f, ensure_ascii=False, indent=2)

    # Also save individual text files for convenience
    book_path = os.path.join(state.output_dir, f"{state.document_name}_book.txt")
    with open(book_path, "w", encoding="utf-8") as f:
        f.write(state.book_content)

    summary_path = os.path.join(state.output_dir, f"{state.document_name}_summary.txt")
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write(state.summary_content)

    return state


class DocumentPreProcessorAgent:
    def __init__(
        self, model_name: Optional[str] = None, output_dir: str = "processed_documents"
    ):
        # Define the processing workflow
        workflow = StateGraph(DocumentProcessorState)

        # Add nodes
        workflow.add_node("split_batches", split_into_batches)
        workflow.add_node("process_batch", process_batch)
        workflow.add_node("save_results", save_results)  # Add the save_results node
        workflow.add_node("handle_error", handle_error)

        # Define the edges
        workflow.add_edge("split_batches", "process_batch")
        workflow.add_conditional_edges(
            "process_batch",
            check_completion,
            {
                "continue": "process_batch",
                "handle_error": "handle_error",
                "save_results": "save_results",
            },
        )
        workflow.add_edge("save_results", END)
        workflow.add_edge("handle_error", END)

        # Set the entrypoint
        workflow.set_entry_point("split_batches")

        # Compile the graph
        self.graph = workflow.compile()
        self.output_dir = output_dir

    def preprocess_document(
        self, document_text: str, document_name: str = "document"
    ) -> Tuple[List[Chunk], List[Chunk], str]:
        """
        Process a document using the Langgraph workflow.

        Args:
            document_text: Text of the document to process
            document_name: Name of the document (used for file naming)

        Returns:
            Tuple of (book_content_chunks, summary_chunks, json_path)
        """
        # Initialize the state
        initial_state = DocumentProcessorState(
            document_text=document_text,
            document_name=document_name,
            output_dir=self.output_dir,
        )

        # Run the workflow
        final_state = self.graph.invoke(initial_state)

        # Create chunks from the results - access dictionary values properly
        book_chunks = [
            Chunk(
                text=final_state["book_content"],
                metadata={
                    "source_type": "book_content",
                    "document_name": document_name,
                },
            )
        ]

        summary_chunks = [
            Chunk(
                text=final_state["summary_content"],
                metadata={"source_type": "summary", "document_name": document_name},
            )
        ]

        # Path to the JSON file
        json_path = os.path.join(self.output_dir, f"{document_name}.json")

        return book_chunks, summary_chunks, json_path

    def preprocess_file(self, file_path: str) -> Tuple[List[Chunk], List[Chunk], str]:
        """
        Process a document file.

        Args:
            file_path: Path to the text file to process

        Returns:
            Tuple of (book_content_chunks, summary_chunks, json_path)
        """
        try:
            document_name = Path(file_path).stem

            with open(file_path, "r", encoding="utf-8") as f:
                document_text = f.read()

            return self.preprocess_document(document_text, document_name)

        except Exception as e:
            print(f"Error processing file: {str(e)}")
            return [], [], ""


# if __name__ == "__main__":

# file_path = "data/the Origin of Species.txt"
# output_dir = "processed_documents"

# processor = DocumentPreProcessorAgent(output_dir=output_dir)
# book_chunks, summary_chunks, json_path = processor.preprocess_file(file_path)

# print("\nProcessing complete!")
# print(
#     f"Book content length: {len(book_chunks[0].text) if book_chunks else 0} characters"
# )
# print(
#     f"Summary content length: {len(summary_chunks[0].text) if summary_chunks else 0} characters"
# )
# print(f"JSON file saved to: {json_path}")
# print("This JSON file can now be used for chunking.")
