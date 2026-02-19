"""
Document Processor
==================
Handles PDF and TXT file ingestion with smart chunking.
"""

import os
import re
import logging
from typing import List, Dict, Any
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class DocumentChunk:
    text: str
    metadata: Dict[str, Any]


class DocumentProcessor:
    """
    Reads PDF and TXT files, cleans text, and splits into
    overlapping chunks optimised for RAG retrieval.
    """

    def __init__(self, chunk_size: int = 500, chunk_overlap: int = 100):
        self.chunk_size    = chunk_size
        self.chunk_overlap = chunk_overlap

    # ── Public API ────────────────────────────────────────────────────────────

    def process_file(self, filepath: str) -> List[DocumentChunk]:
        ext = os.path.splitext(filepath)[1].lower()
        if ext == ".pdf":
            text = self._read_pdf(filepath)
        elif ext == ".txt":
            text = self._read_txt(filepath)
        else:
            raise ValueError(f"Unsupported file type: {ext}")

        text   = self._clean_text(text)
        chunks = self._split_text(text, source=os.path.basename(filepath))
        logger.info(f"Processed '{filepath}' → {len(chunks)} chunks")
        return chunks

    # ── Private helpers ────────────────────────────────────────────────────────

    def _read_pdf(self, filepath: str) -> str:
        try:
            import pypdf
            reader = pypdf.PdfReader(filepath)
            pages  = []
            for i, page in enumerate(reader.pages):
                page_text = page.extract_text() or ""
                if page_text.strip():
                    pages.append(f"[Page {i+1}]\n{page_text}")
            return "\n\n".join(pages)
        except ImportError:
            raise ImportError("Install pypdf: pip install pypdf")
        except Exception as e:
            raise RuntimeError(f"PDF read error: {e}")

    def _read_txt(self, filepath: str) -> str:
        encodings = ["utf-8", "latin-1", "cp1252"]
        for enc in encodings:
            try:
                with open(filepath, "r", encoding=enc) as f:
                    return f.read()
            except UnicodeDecodeError:
                continue
        raise RuntimeError(f"Cannot decode {filepath}")

    def _clean_text(self, text: str) -> str:
        text = re.sub(r'\s+', ' ', text)           # collapse whitespace
        text = re.sub(r'[^\x00-\x7F]+', ' ', text) # strip non-ASCII
        text = re.sub(r'\.{3,}', '…', text)        # ellipsis
        text = text.strip()
        return text

    def _split_text(self, text: str, source: str) -> List[DocumentChunk]:
        """
        Sentence-aware sliding-window chunking.
        Tries to split on sentence boundaries before hard-cutting.
        """
        # Split into sentences first
        sentences = re.split(r'(?<=[.!?])\s+', text)
        
        chunks   = []
        current  = []
        curr_len = 0

        for sentence in sentences:
            s_len = len(sentence)
            if curr_len + s_len > self.chunk_size and current:
                chunk_text = " ".join(current).strip()
                if chunk_text:
                    chunks.append(DocumentChunk(
                        text=chunk_text,
                        metadata={
                            "source":     source,
                            "chunk_idx":  len(chunks),
                            "char_count": len(chunk_text)
                        }
                    ))
                # Overlap: keep last N chars worth of sentences
                overlap_text = chunk_text[-self.chunk_overlap:]
                current  = [overlap_text]
                curr_len = len(overlap_text)

            current.append(sentence)
            curr_len += s_len + 1

        # Flush remaining
        if current:
            chunk_text = " ".join(current).strip()
            if chunk_text:
                chunks.append(DocumentChunk(
                    text=chunk_text,
                    metadata={
                        "source":     source,
                        "chunk_idx":  len(chunks),
                        "char_count": len(chunk_text)
                    }
                ))

        return chunks
