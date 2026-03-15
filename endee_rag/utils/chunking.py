from typing import List
import re

class TextChunker:
    """ Utility for splitting text into overlapping chunks. """

    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 200):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def split_text(self, text: str) -> List[str]:
        """ Splits text into chunks based on size and overlap. """
        # Clean whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        
        chunks = []
        if len(text) <= self.chunk_size:
            return [text]

        start = 0
        while start < len(text):
            end = start + self.chunk_size
            
            # Try to find a good breaking point (period, newline, or space)
            if end < len(text):
                # Search backwards for a sentence end
                period_match = text.rfind('.', start, end + 1)
                if period_match != -1 and period_match > start + (self.chunk_size // 2):
                    end = period_match + 1
                else:
                    # Fallback to space
                    space_match = text.rfind(' ', start, end + 1)
                    if space_match != -1 and space_match > start + (self.chunk_size // 2):
                        end = space_match
            
            chunk = text[start:end].strip()
            if chunk:
                chunks.append(chunk)
            
            # Move start forward by chunk_size - overlap
            start += (self.chunk_size - self.chunk_overlap)
            
            # Ensure we don't get stuck in an infinite loop
            if start >= len(text) - self.chunk_overlap and len(chunks) > 0:
                if start < len(text):
                    last_chunk = text[start:].strip()
                    if last_chunk and last_chunk not in chunks[-1]:
                        chunks.append(last_chunk)
                break
                
        return chunks
