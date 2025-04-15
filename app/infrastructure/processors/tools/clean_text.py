import re
from typing import Any, Dict
from langchain_core.tools import tool


from langchain.tools import tool
import re

@tool
def clean_text(raw_text: str) -> str:
    """
    Cleans text by removing Project Gutenberg headers, artificial line breaks, and spaces.
    """
    start_marker = "*** START OF THE PROJECT GUTENBERG EBOOK"
    if start_marker in raw_text:
        raw_text = raw_text.split(start_marker, maxsplit=1)[-1]

    end_marker = "*** END OF THE PROJECT GUTENBERG EBOOK"
    if end_marker in raw_text:
        raw_text = raw_text.split(end_marker, maxsplit=1)[0]

    cleaned = re.sub(r"\n[A-Z \d,.'\-:;]{5,}\n", "\n", raw_text)
    cleaned = re.sub(r'(?<!\n)\n(?!\n)', ' ', cleaned)
    cleaned = re.sub(r'\n{2,}', '\n\n', cleaned)
    cleaned = "\n".join([line.strip() for line in cleaned.splitlines()])

    return cleaned

# @tool()
# def clean_text(text: str) -> str:
#     """
#     Clean and normalize text by removing unnecessary spaces, newlines, and headers.
#     """

#     start_marker = "*** START OF THE PROJECT GUTENBERG EBOOK"
#     if start_marker in text:
#         text = text.split(start_marker, maxsplit=1)[-1]

#     # 2. Cut content after book end
#     end_marker = "*** END OF THE PROJECT GUTENBERG EBOOK"
#     if end_marker in text:
#         text = text.split(end_marker, maxsplit=1)[0]

#     # 3. Remove uppercase headers and noise
#     cleaned = re.sub(r"\n[A-Z \d,.'\-:;]{5,}\n", "\n", text)

#     # 4. Join artificial line breaks
#     cleaned = re.sub(r'(?<!\n)\n(?!\n)', ' ', cleaned)

#     # 5. Normalize paragraphs
#     cleaned = re.sub(r'\n{2,}', '\n\n', cleaned)

#     # 6. Remove unnecessary spaces
#     cleaned = "\n".join([line.strip() for line in cleaned.splitlines()])
    
#     return cleaned
    
# @tool()
# def extract_sections(self, text: str) -> Dict[str, str]:
#     """
#     Extract chapter/section titles and their content from document text.
#     Returns a dictionary mapping section titles to their content.
#     """
#     # Simple section extraction with regex
#     # Assumes chapters/sections start with "CHAPTER" or roman numerals
#     sections = {}
#     pattern = r"(?:CHAPTER|Chapter)\s+(?:[IVX]+|\d+)[.\s]+(.+?)(?=(?:CHAPTER|Chapter)\s+(?:[IVX]+|\d+)|$)"
#     matches = re.finditer(pattern, text, re.DOTALL)
    
#     for i, match in enumerate(matches):
#         title = f"Chapter {i+1}"
#         content = match.group(1).strip()
#         sections[title] = content
        
#     return sections

# @tool()
# def analyze_document(self, text: str) -> Dict[str, Any]:
#     """
#     Analyze document to identify format, language, structure and potential issues.
#     """
#     result = {
#         "format": "unknown",
#         "language": "unknown",
#         "word_count": len(text.split()),
#         "char_count": len(text),
#         "issues": []
#     }
    
#     # Format detection
#     if "*** START OF THE PROJECT GUTENBERG EBOOK" in text:
#         result["format"] = "project_gutenberg"
    
#     # Language detection (simple heuristic)
#     english_markers = ["the", "and", "of", "to", "in", "that"]
#     spanish_markers = ["el", "la", "los", "en", "y", "que"]
    
#     english_count = sum(1 for word in text.lower().split() if word in english_markers)
#     spanish_count = sum(1 for word in text.lower().split() if word in spanish_markers)
    
#     if english_count > spanish_count:
#         result["language"] = "english"
#     elif spanish_count > english_count:
#         result["language"] = "spanish"
        
#     # Issue detection
#     if "�" in text:
#         result["issues"].append("encoding_errors")
#     if len(text.strip()) == 0:
#         result["issues"].append("empty_document")
        
#     return result
