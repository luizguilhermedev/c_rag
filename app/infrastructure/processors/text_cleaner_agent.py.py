import logging
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from langgraph.prebuilt import create_react_agent
from langchain.prompts import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate
from langchain.text_splitter import RecursiveCharacterTextSplitter

from app.infrastructure.processors.tools.clean_text import clean_text

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

tools = [clean_text]

prompt = ChatPromptTemplate.from_messages([
    ("system", "You are an assistant specialized in text cleaning. Remove headers, footers, and unnecessary line breaks. After cleaning, structure the text efficiently for a RAG or CRAG system."),
    ("placeholder", "{text}")
])

agent_executor = create_react_agent(
    tools=tools,
    model=llm,
    prompt=prompt,
)

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=10000,
    chunk_overlap=100
)

with open("data/the Origin of Species.txt", "r", encoding="utf-8") as f:
    raw_text = f.read()

text_chunks = text_splitter.split_text(raw_text)
logging.info(f"Total batches generated: {len(text_chunks)}")

cleaned_chunks = []
for i, chunk in enumerate(text_chunks):
    logging.info(f"Processing batch {i + 1} of {len(text_chunks)}...")
    
    inputs = {"messages": [("user", f"Please clean the following text: {chunk}")]}
    
    try:
        logging.debug(f"Sending batch {i + 1} for processing")
        result = agent_executor.invoke(inputs)
        
        messages = result["messages"]
        last_ai_message = next((msg for msg in reversed(messages) 
                               if msg.type == "ai" and not msg.tool_calls), None)
        
        if last_ai_message:
            logging.info(f"Batch {i + 1} processed successfully!")
            logging.debug(f"Agent reasoning: {[m.content for m in messages if m.type == 'ai']}")
            cleaned_chunks.append(last_ai_message.content)
        else:
            logging.warning(f"No final response found for batch {i + 1}")
            
    except Exception as e:
        logging.error(f"Error processing batch {i + 1}: {e}")
        continue

# Combine cleaned batches
cleaned_text = "\n\n".join(cleaned_chunks)

# Save the cleaned text
output_file = "clean_origin_of_species.txt"
with open(output_file, "w", encoding="utf-8") as f:
    f.write(cleaned_text)

logging.info(f"Processing completed successfully! Cleaned text saved to {output_file}")