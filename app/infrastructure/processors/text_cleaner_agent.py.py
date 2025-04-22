import logging
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from langgraph.prebuilt import create_react_agent
from langchain.prompts import ChatPromptTemplate
from langchain.text_splitter import RecursiveCharacterTextSplitter

from app.infrastructure.processors.tools.clean_text import preprocess_text

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


llm = ChatOpenAI(model="gpt-4o-mini", temperature=0, request_timeout=300)

tools = [preprocess_text]

# prompt = ChatPromptTemplate.from_messages([
#     ("system", """
#         You are an agent specialized in cleaning and analyzing text documents.
#         Your task is to analyze the text, remove All the irrelevant parts of the book, such as copyrights, headers and
#         any other irrelevant informations not related to the main content of the book. Do not alter the content of the book. Just remove the irrelevant parts.
#         You should also extract the chapters and sections of the book, and rewrite in a way that will be easy to use in a RAG or CRAG system.
#      """),
#     ("placeholder", "{text}")
# ])

# agent_executor = create_react_agent(
#     tools=tools,
#     model=llm,
#     prompt=prompt,
# )


text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=10000,
    chunk_overlap=100
)

with open("data/the Origin of Species.txt", "r", encoding="utf-8") as f:
    raw_text = f.read()

text_chunks = text_splitter.split_text(raw_text)
logging.info(f"Total batches generated: {len(text_chunks)}")

all_cleaned_content = []

for i, chunk in enumerate(text_chunks):
    logging.info(f"Processing chunk {i+1}/{len(text_chunks)}...")
    
    messages = [
        SystemMessage(content="""
            You are specialized in cleaning and analyzing text documents.
            The text is a book, so it may contain chapters and sections.
            Analyze the text, remove all irrelevant parts such as copyrights, headers and
            any other irrelevant information not related to the main content of the book.
            You should also remove summary and indexes, such as Cha´pters, etc....
            You should also remove dedicatories and acknowledgments.
            You should remove trailing spaces and new lines.
            Do not alter the core content of the book. Just remove the irrelevant parts.
            Return only the cleaned text with no explanations or additional commentary.
            Pay attention because it will be used as a RAG or CRAG document.
            The text is in English.
        """),
        HumanMessage(content=f"Clean the following text: {chunk}")
    ]
    
    try:
        response = llm.invoke(messages)
        cleaned_text = response.content
        
        logging.info(f"Chunk {i+1} cleaned successfully")
        logging.info(f"First 200 characters: {cleaned_text[:200].replace(chr(10), '\\n')}")
        
        batch_file = f"data/clean_origin_batch_{i+1}.txt"
        with open(batch_file, "w", encoding="utf-8") as f:
            f.write(cleaned_text)
        logging.info(f"Saved batch {i+1} to {batch_file}")
        
        all_cleaned_content.append(cleaned_text)
    except Exception as e:
        logging.error(f"Error processing chunk {i+1}: {e}")

# Concatenate all cleaned content
if all_cleaned_content:
    complete_cleaned_text = "\n\n".join(all_cleaned_content)
    
    # Save complete cleaned document
    output_file = "data/clean_origin_complete.txt"
    with open(output_file, "w", encoding="utf-8") as f:
        f.write(complete_cleaned_text)
    
    logging.info(f"Complete cleaned document saved to {output_file}")
    
    # Optional: Use preprocessing function on the complete text if needed
    # preprocessed_text = preprocess_text.run(complete_cleaned_text)
else:
    logging.error("No cleaned content was produced")