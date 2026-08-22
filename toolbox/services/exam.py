"""Phase 2 (School Days) Problem Set 1: exam-material recall under a
900-token (o200k_base) budget."""

import json
import logging
import os
import re
import threading
from urllib.parse import urljoin

import requests
import tiktoken

logger = logging.getLogger(__name__)

API_BASE = os.environ.get("TOOLBOX_API_BASE", "https://tool-box-2591eaa24fa3.herokuapp.com")
_ENCODING = None
_FETCH_TIMEOUT = 5
TOKEN_BUDGET = 900
_MAX_PASSAGE_TOKENS = 220

_lock = threading.Lock()
_encoding_lock = threading.Lock()
_passages = None


def _get_encoding():
    global _ENCODING
    if _ENCODING is None:
        with _encoding_lock:
            if _ENCODING is None:
                _ENCODING = tiktoken.get_encoding("o200k_base")
    return _ENCODING


def _fetch_json_or_text(url, **params):
    response = requests.get(url, params=params or None, timeout=_FETCH_TIMEOUT)
    response.raise_for_status()
    try:
        return response.json()
    except ValueError:
        return response.text


def _extract_addresses(listing):
    if isinstance(listing, dict):
        for key in ("materials", "documents", "items", "data"):
            if isinstance(listing.get(key), list):
                listing = listing[key]
                break
    if not isinstance(listing, list):
        raise ValueError("unexpected study-materials listing shape")

    addresses = []
    for entry in listing:
        if isinstance(entry, str):
            addresses.append(entry)
        elif isinstance(entry, dict):
            for key in ("address", "url", "path", "href"):
                if entry.get(key):
                    addresses.append(entry[key])
                    break
    return addresses


def _document_text(payload):
    if isinstance(payload, str):
        return payload
    if isinstance(payload, dict):
        for key in ("content", "text", "body", "material"):
            if isinstance(payload.get(key), str):
                return payload[key]
    return json.dumps(payload)


def _split_long_passage(text, max_tokens=_MAX_PASSAGE_TOKENS):
    sentences = re.split(r"(?<=[.!?])\s+", text)
    chunks = []
    current, current_tokens = [], 0
    for sentence in sentences:
        tokens = len(_get_encoding().encode(sentence))
        if current and current_tokens + tokens > max_tokens:
            chunks.append(" ".join(current))
            current, current_tokens = [], 0
        current.append(sentence)
        current_tokens += tokens
    if current:
        chunks.append(" ".join(current))
    return chunks


def _passages_from_text(text):
    paragraphs = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
    passages = []
    for paragraph in paragraphs:
        if len(_get_encoding().encode(paragraph)) <= _MAX_PASSAGE_TOKENS:
            passages.append(paragraph)
        else:
            passages.extend(_split_long_passage(paragraph))
    return passages


def _load_materials():
    listing = _fetch_json_or_text(urljoin(API_BASE + "/", "study-materials"))
    addresses = _extract_addresses(listing)
    passages = []
    for address in addresses:
        url = urljoin(API_BASE + "/", address)
        try:
            payload = _fetch_json_or_text(url)
        except requests.RequestException:
            logger.warning("failed to fetch study material %s", url)
            continue
        passages.extend(_passages_from_text(_document_text(payload)))
    return passages


def _materials():
    global _passages
    with _lock:
        if _passages is None:
            try:
                _passages = _load_materials()
            except requests.RequestException:
                logger.warning("failed to fetch study-materials listing")
                _passages = []
        return _passages


def _score(question, passage):
    q_words = set(re.findall(r"[a-z0-9]+", question.lower()))
    p_words = set(re.findall(r"[a-z0-9]+", passage.lower()))
    if not q_words or not p_words:
        return 0
    return len(q_words & p_words)


def _select_passages(question, passages):
    ranked = sorted(passages, key=lambda p: _score(question, p), reverse=True)
    selected = []
    total_tokens = 0
    for passage in ranked:
        tokens = len(_get_encoding().encode(passage))
        if total_tokens + tokens > TOKEN_BUDGET:
            continue
        selected.append(passage)
        total_tokens += tokens
    return selected


def recall_study_material(args):
    question = str(args.get("question") or "").strip()
    if not question:
        raise ValueError("question is required")
    passages = _materials()
    if not passages:
        return []
    return _select_passages(question, passages)
