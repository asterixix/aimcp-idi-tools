"""Chunking utilities: semantic chunking for long documents."""
import re
from typing import List
from .utils import enable_function_tracing


def _split_long_block_into_sentences(block: str, target_chars: int) -> List[str]:
    sentences = re.split(r"(?<=[\.\!?。！？])\s+", block.strip())
    if len(sentences) <= 1:
        return [block.strip()]

    pieces = []
    current = []
    current_len = 0

    for sentence in sentences:
        sentence = sentence.strip()
        if not sentence:
            continue
        if current and current_len + len(sentence) + 1 > target_chars:
            pieces.append(" ".join(current).strip())
            current = [sentence]
            current_len = len(sentence)
        else:
            current.append(sentence)
            current_len += len(sentence) + 1

    if current:
        pieces.append(" ".join(current).strip())

    return [piece for piece in pieces if piece]


def _chunk_tail_overlap(text: str, overlap_chars: int) -> str:
    if overlap_chars <= 0 or len(text) <= overlap_chars:
        return text.strip()
    tail = text[-overlap_chars:]
    boundary = tail.find("\n\n")
    if boundary > 0:
        tail = tail[boundary + 2 :]
    return tail.strip()


def build_semantic_chunks(text: str, target_chars: int = 12000, overlap_chars: int = 1500) -> List[dict]:
    normalized = text.replace("\r\n", "\n").replace("\r", "\n").strip()
    normalized = re.sub(r"(?m)^(#{1,6}\s+.+)$", r"\n\1\n", normalized)

    blocks = [block.strip() for block in re.split(r"\n{2,}", normalized) if block.strip()]
    chunk_texts = []
    current_blocks = []
    current_len = 0

    def flush_current():
        nonlocal current_blocks, current_len
        if not current_blocks:
            return
        chunk_text = "\n\n".join(current_blocks).strip()
        if chunk_text:
            chunk_texts.append(chunk_text)
        overlap = _chunk_tail_overlap(chunk_text, overlap_chars)
        current_blocks = [overlap] if overlap else []
        current_len = len(overlap)

    for block in blocks:
        if len(block) > target_chars:
            flush_current()
            sentence_pieces = _split_long_block_into_sentences(block, target_chars)
            for piece in sentence_pieces:
                if len(piece) > target_chars:
                    pieces = [piece[i : i + target_chars] for i in range(0, len(piece), target_chars)]
                    sentence_pieces = pieces
                    break
            for piece in sentence_pieces:
                piece = piece.strip()
                if not piece:
                    continue
                if current_blocks and current_len + len(piece) + 2 > target_chars:
                    flush_current()
                current_blocks.append(piece)
                current_len += len(piece) + 2
            continue

        projected = current_len + len(block) + (2 if current_blocks else 0)
        if current_blocks and projected > target_chars:
            flush_current()

        current_blocks.append(block)
        current_len += len(block) + (2 if len(current_blocks) > 1 else 0)

    if current_blocks:
        chunk_text = "\n\n".join(current_blocks).strip()
        if chunk_text:
            chunk_texts.append(chunk_text)

    chunks = []
    for index, chunk_text in enumerate(chunk_texts, 1):
        first_line = next((line.strip() for line in chunk_text.splitlines() if line.strip()), "")
        section_hint = first_line.lstrip("#").strip() if first_line.startswith("#") else first_line[:120]
        chunks.append({
            "index": index,
            "text": chunk_text,
            "chars": len(chunk_text),
            "section_hint": section_hint or f"chunk_{index}",
            "preview": chunk_text[:240].replace("\n", " "),
        })

    return chunks


enable_function_tracing(globals(), __name__, include_private=True)
