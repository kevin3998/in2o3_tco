import os
import json
import re
import sqlite3
import argparse
import logging
import time
import multiprocessing
from pathlib import Path
from typing import List, Dict, Optional, Any, Tuple, Union
from io import BytesIO

import fitz  # PyMuPDF
import requests  # 新增：用于CrossRef API调用
from tqdm import tqdm

# --- Configuration Constants ---
CONFIG = {
    "CROSSREF_API": {
        "BASE_URL": "https://api.crossref.org/works/",
        "MAILTO": "chenlintao3998@gmail.com"  # <<< IMPORTANT: Set your email here or via args
    },
    "REF_FILTER": {
        "MIN_ENTRIES": 5,
        "POSITION_THRESHOLD": 0.65,
        "TITLE_PATTERNS": [
            r'^\s*references?\b', r'^\s*bibliography\b', r'^\s*works?\s*cited\b',
            r'^\s*references?\s*and\s*notes\b', r'^\s*文献引用\b', r'^\s*参考文献\b',
            r'^\s*cite\s*this\s*article\b', r'^\s*引用文献\b', r'^\s*参照文献\b',
            r'^\s*主要参考文献\b'
        ],
        "ENTRY_PATTERNS": [
            r'^\s*(\[?\d+\]?|•|\*)\s', r'\b\d{4}[a-z]?\b', r'\b(?:pp?\.|pages?)\s+\d+',
            r'\bvol\.?\s+\d+', r'\bno\.?\s+\d+', r'\bdoi:\s*10\.\d+', r'\bISBN\b',
            r'[A-Z][a-z]+,\s[A-Z]\.?,?\s*\d{4}', r'\bin\s+[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b',
            r'\b(?:ed\.|editor|编)\b', r'\bretrieved\s+from\b',
            r'[\(（《【][A-Za-z]+\s*，?\s*\d{4}[\)）》】]'
        ],
        "DENSITY_WINDOW_SIZE": 1000,
        "DENSITY_MAX_WINDOWS": 10,
        "DENSITY_THRESHOLD": 7,
        "TITLE_CANDIDATE_SCORE_THRESHOLD": 2,
        "CONSERVATIVE_TRIM_PERCENT": 0.85,
        "CONSERVATIVE_TRIM_MIN_MATCHES": 8,
    },
    "KEYWORD_EXTRACTOR": {
        "MIN_KEYWORDS": 3, "MAX_KEYWORDS": 8,
        "PATTERNS": [
            (r'(keywords?|key words?|关键词|关键字)\s*[:：]\s*', 0.9),
            (r'\b[A-Z][a-zA-Z]{3,}(?:\s+[A-Z][a-zA-Z]{3,}){1,}\b', 0.5),
            (r'\b(?:\w+[-/]?){2,}(?=\s*[,;])', 0.3)
        ],
        "MAX_TEXT_SEARCH_LENGTH": 4000,
        "MIN_KEYWORD_LEN": 3, "MAX_KEYWORD_LEN": 45,
    },
    "DOI_REGEX": r'\b(10\.\d{4,9}/[-._;()/:A-Z0-9]+)\b',
    "TIMEOUT_PER_PDF": 45,
    "LOG_FILE": "pdf_processing_improved.log",
    "LOG_LEVEL": logging.INFO,
}

# --- Logging Setup (Keep as is) ---
logging.basicConfig(
    level=CONFIG["LOG_LEVEL"],
    format='%(asctime)s - %(levelname)s - [%(funcName)s:%(lineno)d] - %(message)s',
    handlers=[
        logging.FileHandler(CONFIG["LOG_FILE"], mode='a', encoding='utf-8'),
        logging.StreamHandler()
    ]
)


# --- Helper: Text Cleaning ---
def _clean_text(text: Optional[str]) -> str: # Ensure this is your improved version
    if not text: return ""
    text = text.replace('\uFFFD', '')
    text = re.sub(r'[\x00-\x08\x0B\x0C\x0E-\x1F\x7F]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text


# --- Helper: CrossRef API Query ---
def query_crossref_by_doi(doi: str) -> Optional[Dict[str, Any]]:
    if not doi:
        return None
    url = f"{CONFIG['CROSSREF_API']['BASE_URL']}{doi}"
    headers = {
        'User-Agent': f"CrossRefQuery/1.0 (mailto:{CONFIG['CROSSREF_API']['MAILTO']})"
    }
    try:
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()  # Raises an exception for 4XX/5XX errors
        data = response.json()
        if data.get("status") == "ok" and "message" in data:
            logging.info(f"CrossRef: Successfully fetched metadata for DOI: {doi}")
            return data["message"]
        else:
            logging.warning(
                f"CrossRef: Received non-ok status or no message for DOI: {doi}. Status: {data.get('status')}")
    except requests.exceptions.RequestException as e:
        logging.error(f"CrossRef: API request failed for DOI {doi}: {e}")
    except json.JSONDecodeError as e:
        logging.error(f"CrossRef: Failed to decode JSON response for DOI {doi}: {e}")
    return None


# --- ReferenceFilter Class (Refined) ---
class ReferenceFilter:
    def __init__(self, config: Dict[str, Any]):
        self.cfg = config
        self.title_pattern = re.compile(
            '(' + '|'.join(self.cfg["TITLE_PATTERNS"]) + ')',
            flags=re.IGNORECASE | re.MULTILINE
        )
        self.entry_pattern = re.compile(
            '|'.join(self.cfg["ENTRY_PATTERNS"]),
            flags=re.IGNORECASE | re.MULTILINE
        )

    def remove_references(self, text: str) -> str:
        original_len = len(text)
        if original_len == 0: return ""

        pos = self._locate_by_title(text, original_len)
        if pos != -1:
            logging.debug(f"Ref filter: Stage 1 (title) applied at pos {pos}")
            return text[:pos]

        pos = self._detect_citation_density(text, original_len)
        if pos != -1:
            logging.debug(f"Ref filter: Stage 2 (density) applied at pos {pos}")
            return text[:pos]

        pos = self._find_structural_break(text, original_len)  # Added original_len
        if pos != -1:
            logging.debug(f"Ref filter: Stage 3 (structural break) applied at pos {pos}")
            return text[:pos]

        logging.debug("Ref filter: Stages 1-3 no definitive break found, trying conservative trim.")
        return self._conservative_trim(text, original_len)

    def _locate_by_title(self, text: str, text_len: int) -> int:
        candidates = []
        if text_len == 0: return -1

        for match in self.title_pattern.finditer(text):
            start = match.start()
            # Check if title is reasonably near the end and followed by reference-like content
            if (start / text_len) > self.cfg["POSITION_THRESHOLD"]:
                context_after = text[match.end(): min(text_len, match.end() + 500)]
                num_entry_indicators = len(self.entry_pattern.findall(context_after))
                # Score based on position and density of reference entries after title
                score = 2 + num_entry_indicators // 2  # Higher score if more entries follow
                candidates.append((start, score))

        if not candidates: return -1
        best_candidate = max(candidates, key=lambda x: x[1])
        logging.debug(f"Ref locate_by_title candidates: {candidates}, best: {best_candidate}")
        return best_candidate[0] if best_candidate[1] >= self.cfg["TITLE_CANDIDATE_SCORE_THRESHOLD"] else -1

    def _detect_citation_density(self, text: str, text_length: int) -> int:
        if text_length < self.cfg["DENSITY_WINDOW_SIZE"]: return -1

        for i in range(self.cfg["DENSITY_MAX_WINDOWS"]):  # Iterate from the end
            end_pos = text_length - (i * self.cfg["DENSITY_WINDOW_SIZE"] // 2)  # Overlapping windows
            start_pos = max(0, end_pos - self.cfg["DENSITY_WINDOW_SIZE"])
            window = text[start_pos:end_pos]

            if not window: continue
            matches = len(self.entry_pattern.findall(window))
            density = (matches * 1000) / len(window) if len(window) > 0 else 0
            logging.debug(
                f"Ref density check: window {start_pos}-{end_pos}, matches: {matches}, density: {density:.2f}")

            if density >= self.cfg["DENSITY_THRESHOLD"]:
                # Try to find a natural break (e.g., paragraph end) near this point
                actual_break = text.rfind("\n\n", 0,
                                          start_pos + len(window) // 3)  # Search backwards from start of dense region
                return actual_break if actual_break != -1 else start_pos  # Fallback to start_pos
        return -1

    def _find_structural_break(self, text: str, text_len: int) -> int:
        # Simplified, as density and title are primary. This can catch acknowledgements.
        ack_patterns = [
            r'\n\s*acknowledg?e?ments?\b', r'\n\s*致\s*谢\b',
            r'\n\s*author\s*contributions\b', r'\n\s*funding\s*information\b',
            r'\n\s*declaration\s*of\s*competing\s*interest\b'
        ]
        earliest_break = text_len
        for pattern in ack_patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match and (match.start() / text_len) > 0.5:  # Ensure it's in the latter half
                earliest_break = min(earliest_break, match.start())

        return earliest_break if earliest_break < text_len else -1

    def _conservative_trim(self, text: str, original_len: int) -> str:
        # This is a last resort if other methods fail.
        # Only trim if the last part indeed looks like references.
        trim_pos = int(original_len * self.cfg["CONSERVATIVE_TRIM_PERCENT"])
        last_part = text[trim_pos:]

        if len(self.entry_pattern.findall(last_part)) >= self.cfg["CONSERVATIVE_TRIM_MIN_MATCHES"]:
            logging.debug(f"Ref conservative trim: applied at {trim_pos}")
            # Try to find a cleaner break point (e.g., end of a paragraph)
            cleaner_trim_pos = text.rfind("\n", 0, trim_pos)
            return text[
                   :cleaner_trim_pos] if cleaner_trim_pos != -1 and cleaner_trim_pos > original_len * 0.5 else text[
                                                                                                               :trim_pos]
        return text


# --- KeywordExtractor Class (Largely similar to your version, minor tweaks) ---
class KeywordExtractor:
    def __init__(self, config: Dict[str, Any]):
        self.cfg = config
        # Ensure patterns are compiled, and correctly handle tuple matches if findall returns them
        self.patterns = []
        for p_text, confidence in self.cfg["PATTERNS"]:
            try:
                self.patterns.append((re.compile(p_text, re.IGNORECASE | re.UNICODE), confidence))
            except re.error as e:
                logging.error(f"KeywordExtractor: Invalid regex pattern '{p_text}': {e}")

    def extract_keywords(self, text: str, title: str) -> List[str]:
        keywords = []
        # Strategy 1: Metadata keywords (usually most reliable)
        meta_keywords = self._get_metadata_keywords(text[:self.cfg["MAX_TEXT_SEARCH_LENGTH"]])  # Search near beginning
        if meta_keywords:
            keywords.extend(meta_keywords)
        keywords = self._unique_preserve_order(keywords)  # Deduplicate early

        # Strategy 2: Pattern-based extraction if not enough keywords
        if len(keywords) < self.cfg["MIN_KEYWORDS"]:
            pattern_keywords = self._extract_by_pattern(text[:self.cfg["MAX_TEXT_SEARCH_LENGTH"]])
            keywords.extend(pattern_keywords)
            keywords = self._unique_preserve_order(keywords)

        # Strategy 3: Title-based extraction if still not enough
        if len(keywords) < self.cfg["MIN_KEYWORDS"] and title:
            title_keywords = self._extract_from_title(title)
            keywords.extend(title_keywords)
            keywords = self._unique_preserve_order(keywords)

        return self._post_process(keywords)

    def _get_metadata_keywords(self, text: str) -> List[str]:
        # More robust pattern for Keywords section
        meta_pattern = r'\b(?:Keywords|Key words|关键词|关键字)\s*[:：]\s*([^\n\r]+(?:\n\r?(?!\s*(?:1\.|I\.|Introduction|摘要|Abstract))[^\n\r]+)*)'
        match = re.search(meta_pattern, text, re.IGNORECASE | re.UNICODE)
        if match and match.group(1):
            return self._split_keywords(match.group(1).strip())
        return []

    def _extract_by_pattern(self, text: str) -> List[str]:
        candidates_with_confidence: List[Tuple[str, float]] = []
        for pattern, confidence in self.patterns:
            try:
                matches = pattern.findall(text)
                for match_item in matches:
                    # Handle cases where findall returns tuples (e.g., if pattern has capturing groups)
                    actual_match = match_item[0] if isinstance(match_item, tuple) and match_item else match_item
                    if isinstance(actual_match, str) and actual_match.strip():
                        # Some patterns might directly give a list of keywords (e.g. if the regex itself handles splitting)
                        # For simplicity, we assume most patterns give a single keyword or a string to be split
                        kw_list_str = actual_match.strip()
                        # Here, we don't split yet, add the matched string with its confidence
                        candidates_with_confidence.append((kw_list_str, confidence))
            except re.error as e:
                logging.warning(f"KeywordExtractor: Regex error during pattern search: {e}")
            except Exception as ex:
                logging.warning(f"KeywordExtractor: Unexpected error during pattern search: {ex}")

        # Sort by confidence, then by length (longer preferred for same confidence)
        # Deduplicate based on the string before splitting
        unique_candidates = self._unique_preserve_order_with_confidence(candidates_with_confidence)

        extracted_keywords: List[str] = []
        for kw_text, conf in sorted(unique_candidates, key=lambda x: (-x[1], -len(x[0]))):
            extracted_keywords.extend(self._split_keywords(kw_text))
            if len(self._unique_preserve_order(extracted_keywords)) >= self.cfg[
                "MAX_KEYWORDS"] * 2:  # Allow more candidates before final trim
                break
        return self._unique_preserve_order(extracted_keywords)

    def _unique_preserve_order_with_confidence(self, items_with_confidence: List[Tuple[str, float]]) -> List[
        Tuple[str, float]]:
        seen_text_lower = set()
        result = []
        for text, confidence in items_with_confidence:
            lower_text = text.lower()
            if lower_text not in seen_text_lower:
                seen_text_lower.add(lower_text)
                result.append((text, confidence))
        return result

    def _extract_from_title(self, title: str) -> List[str]:
        # Simple noun phrase like extraction (can be improved with NLP tools like spaCy/NLTK for POS tagging)
        # This regex looks for sequences of capitalized words, or capitalized words followed by lowercase words
        phrases = re.findall(r'\b[A-Z][a-zA-Z]*(?:\s+(?:[a-z]+|[A-Z][a-zA-Z]*)){0,3}\b', title)
        # Filter out very short phrases or phrases that are likely just names if they are too common
        meaningful_phrases = [p.strip() for p in phrases if len(p.strip().split()) > 1 or len(p.strip()) > 4]
        return self._unique_preserve_order(meaningful_phrases)[:self.cfg["MAX_KEYWORDS"] // 2]  # Take fewer from title

    def _split_keywords(self, keyword_str: str) -> List[str]:
        # More comprehensive separators
        separators = r'[;；,，、\n\r]+|\band\b|\bor\b|\s{2,}|•|·|\|'  # Added 'and', 'or', multiple spaces, pipe
        # Split, strip, and filter empty strings
        return [kw.strip() for kw in re.split(separators, keyword_str) if kw and kw.strip()]

    def _unique_preserve_order(self, items: List[str]) -> List[str]:
        seen = set()
        return [x for x in items if not (x.lower() in seen or seen.add(x.lower()))]

    def _post_process(self, keywords: List[str]) -> List[str]:
        processed = []
        seen_lower = set()
        for kw_candidate in keywords:
            # kw can be a single keyword or a phrase to be split again if missed by initial split
            for kw_part in self._split_keywords(kw_candidate):  # Re-split just in case
                kw_stripped = kw_part.strip(" .:")  # Remove trailing/leading dots, spaces, colons
                # Basic check for meaningful content (e.g., contains at least one letter or CJK char)
                if self.cfg["MIN_KEYWORD_LEN"] <= len(kw_stripped) <= self.cfg["MAX_KEYWORD_LEN"] and \
                        re.search(r'[a-zA-Z\u4e00-\u9fa5]', kw_stripped) and \
                        not kw_stripped.isdigit() and \
                        kw_stripped.lower() not in ["abstract", "keywords", "introduction", "results", "discussion",
                                                    "conclusion", "references"]:  # Exclude common section titles

                    lower_kw = kw_stripped.lower()
                    if lower_kw not in seen_lower:
                        seen_lower.add(lower_kw)
                        processed.append(kw_stripped)  # Keep original casing
        return processed[:self.cfg["MAX_KEYWORDS"]]


# --- Metadata Helper Functions (Refined) ---
def _extract_year_from_metadata(metadata: Dict[str, Any], crossref_data: Optional[Dict[str, Any]] = None) -> Optional[
    str]:
    if crossref_data and 'issued' in crossref_data and 'date-parts' in crossref_data['issued']:
        try:
            return str(crossref_data['issued']['date-parts'][0][0])
        except (IndexError, TypeError, KeyError):
            pass  # Fallback if date-parts is not as expected

    date_keys = ["creationDate", "modDate", "date"]  # 'date' is common in PDF metadata
    for key in date_keys:
        date_str = metadata.get(key, "")
        if isinstance(date_str, str):
            # Try to match YYYY format in various positions
            match = re.search(r'\b(19\d{2}|20\d{2})\b', date_str)
            if match:
                return match.group(1)
            if date_str.startswith("D:"):  # PDF "D:" format
                year_part = date_str[2:6]
                if year_part.isdigit() and len(year_part) == 4:
                    return year_part
    return None


def _parse_authors(author_str: Optional[str], crossref_data: Optional[Dict[str, Any]] = None) -> List[str]:
    if crossref_data and "author" in crossref_data and isinstance(crossref_data["author"], list):
        authors = []
        for auth_entry in crossref_data["author"]:
            if isinstance(auth_entry, dict):
                name_parts = []
                if "given" in auth_entry: name_parts.append(auth_entry["given"])
                if "family" in auth_entry: name_parts.append(auth_entry["family"])
                if name_parts: authors.append(" ".join(name_parts))
        if authors: return authors

    if not author_str or not isinstance(author_str, str): return []
    # More robust splitting, handles "and", "et al.", common academic separators
    # Preserve order if possible, try common delimiters first
    delimiters = r"\s*;\s*|\s*,\s*(?!(?:[JS]r\.|[IVX]+)\b)|\s+\band\b\s+(?![A-Z]\.\s)|\s*和\s*|\s*与\s*"
    authors_list = re.split(delimiters, author_str.strip(), flags=re.IGNORECASE)

    cleaned_authors = []
    for author in authors_list:
        author = author.strip()
        # Remove common affiliations or titles often mixed in
        author = re.sub(r'\s*\(.*?\)\s*$', '', author)  # Remove trailing parenthesized content
        author = re.sub(r'\s*\[.*?\]\s*$', '', author)  # Remove trailing bracketed content
        author = re.sub(r'\s*et al\.?\s*$', '', author, flags=re.IGNORECASE).strip()  # Remove 'et al.'
        author = re.sub(r'^\d+\s*', '', author)  # Remove leading numbers (affiliations)
        author = re.sub(r'\s*\*\s*$', '', author)  # Remove trailing asterisk

        if author and len(author) > 1 and not author.lower() == "others":  # Avoid "others"
            cleaned_authors.append(author)
    return [a for a in cleaned_authors if a]


def _detect_journal_from_text(doc: fitz.Document, crossref_data: Optional[Dict[str, Any]] = None) -> str:
    # Priority 1: CrossRef data
    if crossref_data:
        if "container-title" in crossref_data and crossref_data["container-title"]:
            # CrossRef often provides a list for container-title, take the first
            cr_journal = crossref_data["container-title"]
            if isinstance(cr_journal, list) and cr_journal:
                return cr_journal[0]
            elif isinstance(cr_journal, str):
                return cr_journal
        if "short-container-title" in crossref_data and crossref_data["short-container-title"]:
            cr_short_journal = crossref_data["short-container-title"]
            if isinstance(cr_short_journal, list) and cr_short_journal:
                return cr_short_journal[0]
            elif isinstance(cr_short_journal, str):
                return cr_short_journal

    # Priority 2: Heuristics from PDF text (your existing logic can be a fallback)
    # This function needs to be your existing _detect_journal, adapted.
    # For brevity, I'll assume your _detect_journal function is here and call it.
    # return _your_original_detect_journal_logic(doc)
    # Placeholder for your existing heuristic journal detection
    # Your previous _detect_journal was quite complex, integrate it here.
    # For now, simplified:
    journal_candidates: Dict[str, float] = {}
    # (Insert your refined regex patterns for journal detection here from previous script)
    # Example simplified patterns:
    patterns_confidence = [
        (re.compile(r"^\s*([A-Z][A-Za-z\s&.-]+Journal(?:(?:\s+of\s+|\s+)[\w\s&.-]+)?)\s*$", re.MULTILINE), 5.0),
        (re.compile(r"\b(?:J\.|J\s+|\bJournal\s+of\b|Trans\.|Proc\.)\s*[\w\s&.-]+?(?=\n|$|,|\s\d{4})", re.IGNORECASE),
         3.0),
    ]
    for page_num in range(min(3, len(doc))):  # Check first 3 pages
        page = doc[page_num]
        text_first_half_page = page.get_text("text", clip=fitz.Rect(0, 0, page.rect.width, page.rect.height * 0.5))
        for pattern, confidence in patterns_confidence:
            for match in pattern.finditer(text_first_half_page):
                candidate = (match.group(1) if match.groups() else match.group(0)).strip(" .,")
                if 5 < len(candidate) < 150 and not candidate.lower().startswith(
                        ("abstract", "introduction", "doi:", "issn", "isbn")):
                    journal_candidates[candidate] = journal_candidates.get(candidate, 0) + confidence

    if journal_candidates:
        return max(journal_candidates, key=journal_candidates.get)
    return ""


def _doi_prefix_to_publisher(doi: Optional[str]) -> Optional[str]:
    if not doi: return None
    # More comprehensive map (can be expanded or loaded from a file)
    doi_publisher_map = {
        "10.1016": "Elsevier", "10.1021": "ACS Publications", "10.1039": "Royal Society of Chemistry",
        "10.1002": "Wiley", "10.3390": "MDPI", "10.1080": "Taylor & Francis",
        "10.1007": "Springer Nature", "10.1149": "The Electrochemical Society (ECS)",
        "10.1103": "American Physical Society (APS)", "10.1063": "AIP Publishing",
        "10.1109": "IEEE", "10.1088": "IOP Publishing", "10.1177": "SAGE Publications",
        "10.1000": "International DOI Foundation (IDF) Content",  # General / Test
        "10.1371": "Public Library of Science (PLoS)",
        "10.1186": "BioMed Central (BMC) - part of Springer Nature",
        "10.1246": "The Chemical Society of Japan",
        "10.1073": "National Academy of Sciences (PNAS)",
        "10.1126": "American Association for the Advancement of Science (AAAS Science)",
        "10.1038": "Nature Publishing Group (part of Springer Nature)",
    }
    # Check for most specific match first (e.g., 10.xxxx/yyyy then 10.xxxx)
    if '/' in doi:
        prefix_parts = doi.split('/')
        current_prefix = prefix_parts[0]  # e.g., 10.xxxx
        if current_prefix in doi_publisher_map:
            return doi_publisher_map[current_prefix]
        # More complex matching for sub-prefixes can be added if the map supports it
    return None


def _postprocess_journal_name(journal_name: str, crossref_data: Optional[Dict[str, Any]] = None) -> str:
    if crossref_data:  # Prefer CrossRef's full title if available
        if "container-title" in crossref_data and crossref_data["container-title"]:
            cr_journal = crossref_data["container-title"]
            name = cr_journal[0] if isinstance(cr_journal, list) and cr_journal else str(cr_journal)
            return name.strip()

    if not journal_name: return "Unknown Journal"
    # General cleaning
    name = re.sub(r"^[^\w\u4e00-\u9fa5]+|[^\w\u4e00-\u9fa5]+$", "",
                  journal_name)  # Remove leading/trailing non-alphanum
    name = re.sub(r'\s{2,}', ' ', name).strip()  # Normalize multiple spaces
    # Add more specific abbreviation expansions if needed, or rely on CrossRef
    return name if name else "Unknown Journal"


def _generate_pseudo_doi(pdf_path: Path) -> str:
    try:
        mtime = pdf_path.stat().st_mtime
        fsize = pdf_path.stat().st_size
    except OSError:
        mtime = time.time()
        fsize = 0
    file_hash_part = hash(f"{pdf_path.name}_{fsize}") & 0xFFFFFFFF  # Add size to hash
    return f"local.{int(mtime)}.{fsize}.{file_hash_part:x}"


def _remove_keyword_section_from_text(text: str) -> str:
    # Improved pattern to better handle multiline keyword sections and various terminators
    keyword_section_pattern = re.compile(
        r'\n\s*(?:Keywords|Key words|关键词|关键字|Index Terms)\s*[:：].*?'
        r'(?=\n\s*(?:1\.(?!\d)|I\.(?![VXML])|Introduction|Background|Results|Discussion|Conclusion|本文|引言|摘要|Abstract\b.*\n\n)|$)',
        # Look for common next section headers or end of text
        re.DOTALL | re.IGNORECASE | re.UNICODE
    )
    return keyword_section_pattern.sub('', text)


def _detect_title_from_text(text: str, pdf_metadata_title: Optional[str] = None,
                            crossref_data: Optional[Dict[str, Any]] = None) -> str:
    # Priority 1: CrossRef title
    if crossref_data and "title" in crossref_data and crossref_data["title"]:
        cr_title = crossref_data["title"]
        # CrossRef titles are often in a list, take the first
        title_str = cr_title[0] if isinstance(cr_title, list) and cr_title else str(cr_title)
        return _clean_text(title_str.strip())  # Clean it up

    # Priority 2: PDF Metadata title
    if pdf_metadata_title and len(pdf_metadata_title) > 10:  # Basic check for meaningful metadata title
        # Further clean PDF metadata title (often has artifacts)
        cleaned_meta_title = re.sub(r'\.pdf$', '', pdf_metadata_title, flags=re.IGNORECASE).strip()
        cleaned_meta_title = _clean_text(cleaned_meta_title)
        if len(cleaned_meta_title) > 10:  # Check again after cleaning
            return cleaned_meta_title

    # Priority 3: Heuristic extraction from text (your existing logic refined)
    if not text: return "Unknown Title"

    # Search in the first ~3000 characters (approx. 1-2 pages)
    search_area = text[:3000]
    lines = [line.strip() for line in search_area.split('\n') if line.strip()]

    potential_title = ""
    if not lines: return "Unknown Title"

    # Try to find title in the first few non-empty lines
    # Stop if a line looks like author list, affiliation, abstract keyword, or is too long
    for i, line in enumerate(lines[:10]):  # Check up to 10 lines
        if len(line) > 250 or \
                re.search(r"^(?:abstract|摘要|keywords|关键词|doi:|10\.\d{4,9}/.*)", line, re.IGNORECASE) or \
                re.search(r"\b(?:university|institute|department|college|school|ltd\.|inc\.)\b", line, re.IGNORECASE) or \
                re.search(r"\b[A-Z][a-z]+,\s*[A-Z]\.(?:[A-Z]\.)?", line) or \
                re.search(r"\b(?:[a-z0-9._%+-]+@[a-z0-9.-]+\.[a-z]{2,})\b", line, re.IGNORECASE):  # email
            break

        # If current line is significantly shorter than previous, it might be end of title
        if potential_title and len(line) < len(potential_title.split('\n')[-1]) * 0.6 and len(
                potential_title.split('\n')) > 0:
            break

        potential_title = f"{potential_title}\n{line}" if potential_title else line

        # If next line looks like a clear separator or start of new section
        if i + 1 < len(lines):
            next_line_lower = lines[i + 1].lower()
            if next_line_lower.startswith(("abstract", "摘要", "keywords", "关键词")) or \
                    re.match(r"^(?:[a-z0-9._%+-]+@[a-z0-9.-]+\.[a-z]{2,})$", next_line_lower) or \
                    re.match(r"^(?:https?://)", next_line_lower) or \
                    (len(lines[i + 1]) > 10 and lines[i + 1].isupper()):  # All caps line (often affiliation)
                break
        if len(potential_title) > 250: break  # Safety break for very long titles

    potential_title = _clean_text(potential_title.strip().replace("\n", " "))
    return potential_title if len(potential_title) > 5 else "Unknown Title"


def _extract_abstract_from_text(text: str, crossref_data: Optional[Dict[str, Any]] = None) -> str:
    # Priority 1: CrossRef abstract
    if crossref_data and "abstract" in crossref_data:
        # CrossRef abstract often has XML tags like <jats:p>
        abstract_html = crossref_data["abstract"]
        # Basic HTML tag removal
        abstract_text = re.sub(r'<[^>]+>', ' ', abstract_html)
        return _clean_text(abstract_text.strip())

    # Priority 2: Heuristic extraction from text
    if not text: return ""

    abstract_text = ""
    # Regex for "Abstract" or "摘要" possibly followed by a newline or special characters, then the content
    # It looks for a clear start and tries to capture content until a common next section or too many newlines
    abstract_match = re.search(
        r'\b(?:A\s*B\s*S\s*T\s*R\s*A\s*C\s*T|Abstract|SUMMARY|摘要)\b\s*[:：.]?\s*\n*(.*?)(?=\n\s*(?:Keywords|Key words|关键词|Index Terms|1\.\s*Introduction|I\.\s*Introduction|1\.\s*引言|I\.\s*引言|\n\s*\n\s*\n|Introduction\s*\n\s*\n))',
        text,
        re.IGNORECASE | re.DOTALL | re.UNICODE
    )

    if abstract_match:
        abstract_text = abstract_match.group(1).strip()
        # Further clean to remove potential subsection titles if abstract is very long
        if len(abstract_text) > 2500:  # If abstract is unusually long, might have included keywords
            abstract_text = abstract_text[:2500]  # Cap length
            kwd_match = re.search(r'\n\s*(?:Keywords|Key words|关键词|Index Terms)\s*[:：]', abstract_text,
                                  re.IGNORECASE)
            if kwd_match:
                abstract_text = abstract_text[:kwd_match.start()]
    else:  # Fallback: if no clear "Abstract" keyword, take a chunk after title if possible
        logging.debug("Abstract keyword not found, attempting fallback based on structure.")
        # This is very heuristic and might not be accurate
        # Try to find text after typical title/author block, before intro
        intro_match = re.search(r'\n\s*(?:1\.\s*Introduction|I\.\s*Introduction|1\.\s*引言|I\.\s*引言)', text,
                                re.IGNORECASE)
        if intro_match:
            # Look for a block of text before introduction, likely after authors/affiliations
            # This needs a good way to estimate where authors/affiliations end.
            # For now, this fallback is less reliable and omitted for simplicity to avoid bad extractions.
            pass

    return _clean_text(abstract_text)


def _find_doi_in_text(text: str) -> Optional[str]:
    # Improved DOI regex to be more specific and avoid false positives
    # Handles various DOI formats, including those with special characters in suffix
    doi_pattern = re.compile(CONFIG["DOI_REGEX"], re.IGNORECASE)

    # Search in common places first: first few pages, last few pages
    text_len = len(text)
    search_segments = [text[:5000]]  # Beginning of document
    if text_len > 10000:
        search_segments.append(text[-5000:])  # End of document
    else:  # If document is short, this might overlap, which is fine
        search_segments.append(text)

    for segment in search_segments:
        match = doi_pattern.search(segment)
        if match:
            doi = match.group(1).strip().rstrip('.')  # Remove trailing dots
            # Validate DOI structure somewhat (e.g., not ending with invalid chars)
            if not doi.endswith(('/', '-', '_', ';')):
                logging.info(f"Found DOI in text: {doi}")
                return doi
    logging.debug("No DOI found in common text locations.")
    return None


# --- Main Processing Function (process_pdf_entry - Refined) ---
def process_pdf_entry(
        pdf_data_tuple: Union[Tuple[str], Tuple[Optional[str], ...]],
        ref_filter: ReferenceFilter,
        keyword_extractor: KeywordExtractor
) -> Optional[Dict[str, Any]]:
    # Unpack tuple based on expected length
    pdf_path_str: str
    db_title, db_doi, db_abstract, db_year, db_journal, db_pdf_url = (None,) * 6
    source_type: str

    if len(pdf_data_tuple) == 7:
        db_title, db_doi, db_abstract, db_year, db_journal, db_pdf_url, pdf_path_str = pdf_data_tuple
        source_type = "database_entry"
    elif len(pdf_data_tuple) == 1:
        pdf_path_str = pdf_data_tuple[0]
        source_type = "local_folder"
    else:
        logging.error(f"Invalid pdf_data_tuple format: {pdf_data_tuple}. Expected 1 or 7 elements.")
        return None

    pdf_path = Path(pdf_path_str).resolve()
    log_prefix = f"PDF: {pdf_path.name} -"

    try:
        if not pdf_path.exists() or not pdf_path.is_file():
            logging.warning(f"{log_prefix} File does not exist or is not a file. Path: {pdf_path}")
            return None
        if pdf_path.stat().st_size < 10240:  # Increased min size to 10KB
            logging.warning(f"{log_prefix} File too small (<10KB), likely not a full paper. Path: {pdf_path}")
            return None

        doc = None  # Initialize doc to None
        with fitz.open(pdf_path) as doc:
            if doc.is_encrypted:
                logging.warning(f"{log_prefix} PDF is encrypted. Path: {pdf_path}")
                return None

            pdf_metadata = doc.metadata or {}

            # --- Text Extraction ---
            text_blocks = []
            for page_num, page in enumerate(doc):
                try:
                    flags = fitz.TEXT_PRESERVE_LIGATURES | fitz.TEXT_DEHYPHENATE | fitz.TEXT_MEDIABOX_CLIP
                    page_text = page.get_text("text", flags=flags)
                    text_blocks.append(page_text)
                except Exception as page_e:
                    logging.warning(
                        f"{log_prefix} Error extracting text from page {page_num + 1}: {page_e}. Trying basic.")
                    try:
                        text_blocks.append(page.get_text("text"))  # Fallback
                    except Exception as page_ef:
                        logging.error(
                            f"{log_prefix} Basic text extraction also failed for page {page_num + 1}: {page_ef}")
                        text_blocks.append("")

            raw_full_text = "\n".join(text_blocks).strip()
            if not raw_full_text:
                logging.warning(f"{log_prefix} No text extracted from PDF.")
                return None

            # --- DOI and CrossRef Metadata ---
            final_doi = db_doi  # Prioritize DB DOI
            if not final_doi:  # If no DB DOI, try to find in PDF metadata or text
                final_doi = pdf_metadata.get("doi", None)
                if isinstance(final_doi, bytes):  # Sometimes DOI in metadata is bytes
                    final_doi = final_doi.decode('utf-8', errors='ignore')
                if not final_doi or not re.match(CONFIG["DOI_REGEX"],
                                                 str(final_doi)):  # Validate if it looks like a DOI
                    final_doi = _find_doi_in_text(
                        raw_full_text[:10000] + raw_full_text[-5000:])  # Search beginning and end

            crossref_data: Optional[Dict[str, Any]] = None
            if final_doi:
                final_doi = re.match(CONFIG["DOI_REGEX"], final_doi.strip())  # Clean and validate found DOI
                if final_doi:
                    final_doi = final_doi.group(1)
                    crossref_data = query_crossref_by_doi(final_doi)
                else:  # If regex match failed on supposed DOI
                    final_doi = None

                    # --- Populate Metadata (CrossRef > DB > PDF Meta > Heuristic) ---
            title = db_title or (pdf_metadata.get("title", "") if not crossref_data else None)
            final_title = _detect_title_from_text(raw_full_text, title, crossref_data)

            year = db_year or _extract_year_from_metadata(pdf_metadata, crossref_data)

            authors_list = _parse_authors(pdf_metadata.get("author", ""), crossref_data)

            journal_heuristic = _detect_journal_from_text(doc, crossref_data)  # Pass doc here
            journal = db_journal or journal_heuristic
            final_journal = _postprocess_journal_name(journal, crossref_data)

            if final_journal == "Unknown Journal" and final_doi:  # Try publisher from DOI as last resort
                publisher_from_doi = _doi_prefix_to_publisher(final_doi)
                if publisher_from_doi:
                    final_journal = f"Publisher: {publisher_from_doi}"

            abstract_heuristic = _extract_abstract_from_text(raw_full_text, crossref_data)
            final_abstract = db_abstract or abstract_heuristic
            final_abstract_cleaned = _clean_text(final_abstract)  # Clean after all selections

            # --- Content Processing ---
            text_no_refs = ref_filter.remove_references(raw_full_text)
            extracted_keywords = keyword_extractor.extract_keywords(
                text_no_refs[:CONFIG["KEYWORD_EXTRACTOR"]["MAX_TEXT_SEARCH_LENGTH"]], final_title)  # Use no_refs text

            text_no_keywords_section = _remove_keyword_section_from_text(
                text_no_refs)  # Remove keyword section after extraction
            main_content_cleaned = _clean_text(text_no_keywords_section)

            # Construct LLM-ready text
            llm_ready_parts = [f"Title: {final_title}"]
            if year: llm_ready_parts.append(f"Year: {year}")
            if final_journal != "Unknown Journal": llm_ready_parts.append(f"Journal: {final_journal}")
            if authors_list: llm_ready_parts.append(f"Authors: {'; '.join(authors_list)}")
            if extracted_keywords: llm_ready_parts.append(f"Keywords: {', '.join(extracted_keywords)}")
            if final_abstract_cleaned: llm_ready_parts.append(f"\nAbstract:\n{final_abstract_cleaned}")
            llm_ready_parts.append(f"\nMain Content:\n{main_content_cleaned}")

            llm_ready_fulltext = "\n".join(llm_ready_parts)

            # If still no DOI, generate pseudo DOI
            final_doi = final_doi or _generate_pseudo_doi(pdf_path)

            return {
                "doi": final_doi, "filename": pdf_path.name, "local_path": str(pdf_path),
                "source_type": source_type, "retrieved_title": final_title,
                "retrieved_year": year or "Unknown Year", "retrieved_journal": final_journal,
                "retrieved_authors": authors_list,  # Changed from pdf_meta_authors
                "extracted_keywords": extracted_keywords,
                "extracted_abstract_cleaned": final_abstract_cleaned,
                "llm_ready_fulltext_cleaned": llm_ready_fulltext,
                "raw_fulltext_char_count": len(raw_full_text),
                "norefs_fulltext_char_count": len(text_no_refs),
                "processing_timestamp_utc": time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime()),
                "pdf_metadata_original": {k: str(v) for k, v in pdf_metadata.items() if v is not None},
                # Store original PDF metadata
                "crossref_data_retrieved": bool(crossref_data),  # Flag if CrossRef data was used
                "db_pdf_url": db_pdf_url  # Retain if from DB
            }


    except fitz.fitz.FitzError as fe:
        logging.error(f"{log_prefix} PyMuPDF (Fitz) error: {str(fe)}. Path: {pdf_path}")
        return None  # Return None on specific Fitz errors
    except Exception as e:
        logging.error(f"{log_prefix} Unexpected error in process_pdf_entry: {str(e)}. Path: {pdf_path}", exc_info=True)
        return None  # Return None on other exceptions

# --- Multiprocessing Wrapper (run_with_timeout - Largely similar) ---
def run_with_timeout(
        func_to_run: callable,
        args_tuple: tuple = (),
        timeout_seconds: int = CONFIG["TIMEOUT_PER_PDF"]
) -> Optional[Any]:
    # Use a more robust way to get pdf_path_info if available
    pdf_path_info = "N/A"
    if args_tuple:
        first_arg = args_tuple[0]  # This is pdf_data_tuple
        if isinstance(first_arg, tuple) and first_arg:
            # For DB input, path is last element; for folder input, path is first (and only) element
            pdf_path_info = first_arg[-1] if len(first_arg) == 7 else first_arg[0]
        elif isinstance(first_arg, str):  # Should not happen with current args_tuple structure for process_pdf_entry
            pdf_path_info = first_arg

    # Consider 'fork' for Unix-like, 'spawn' for Windows/macOS if issues arise,
    # but 'spawn' is generally safer for cross-platform compatibility.
    ctx_method = 'spawn' if os.name == 'nt' else 'fork'  # Basic platform check
    try:
        ctx = multiprocessing.get_context(ctx_method)
    except ValueError:  # Fallback if specific method not available (rare)
        logging.warning(f"Multiprocessing context method '{ctx_method}' not available, using default.")
        ctx = multiprocessing.get_context(None)

    with ctx.Pool(processes=1) as pool:
        async_result = pool.apply_async(func_to_run, args=args_tuple)
        try:
            return async_result.get(timeout=timeout_seconds)
        except multiprocessing.TimeoutError:
            logging.warning(
                f"Function {func_to_run.__name__} timed out after {timeout_seconds}s for PDF: {pdf_path_info}")
            return None
        except Exception as e:
            logging.error(f"Exception in child process for {func_to_run.__name__} with PDF {pdf_path_info}: {e}",
                          exc_info=False)  # exc_info=False to prevent massive logs from child
            return None


# --- Main data loading functions (load_from_db, load_from_folder - largely similar logic but pass new config) ---
def load_from_db(db_path_str: str) -> List[Dict[str, Any]]:
    db_path = Path(db_path_str)
    if not db_path.exists():
        logging.error(f"Database file not found: {db_path}")
        return []

    # Initialize filter and extractor once
    ref_filter = ReferenceFilter(CONFIG["REF_FILTER"])
    keyword_extractor = KeywordExtractor(CONFIG["KEYWORD_EXTRACTOR"])
    processed_papers: List[Dict[str, Any]] = []

    try:
        with sqlite3.connect(db_path) as conn:
            cursor = conn.cursor()
            # Ensure the query matches your DB schema and only selects processable entries
            query = """
                SELECT title, doi, abstract, year, journal, pdf_url, save_path 
                FROM literature 
                WHERE save_path IS NOT NULL AND save_path != ''
                ORDER BY RANDOM() 
            """  # Using RANDOM() for better testing diversity if re-running
            try:
                cursor.execute(
                    f"SELECT COUNT(*) FROM ({query.replace('ORDER BY RANDOM()', '')})")  # Count without order by for speed
                total_count = cursor.fetchone()[0]
            except sqlite3.OperationalError as oe:
                logging.error(f"SQLite error during count: {oe}. Trying to proceed without total count.")
                total_count = 0  # Cannot determine total count easily

            if total_count == 0 and not list(cursor.execute(query)):  # Check if truly empty
                logging.info("No processable entries found in the database.")
                return []

            logging.info(
                f"Found {total_count if total_count > 0 else 'an unknown number of'} processable entries in DB.")

            # Re-execute query if total_count was determined, or iterate directly
            db_iterator = cursor.execute(query)

            pbar_desc = "Processing PDFs from DB"
            pbar_total = total_count if total_count > 0 else None  # tqdm handles None total

            success_count = 0
            for row_tuple in tqdm(db_iterator, desc=pbar_desc, total=pbar_total, unit="pdf"):
                pdf_path_in_db = row_tuple[-1]  # save_path
                logging.info(f"Attempting to process from DB: {pdf_path_in_db}")

                result = run_with_timeout(
                    process_pdf_entry,
                    args_tuple=(row_tuple, ref_filter, keyword_extractor)
                )
                if result:
                    processed_papers.append(result)
                    success_count += 1
                    logging.info(f"Successfully processed from DB: {pdf_path_in_db}")
                else:
                    logging.warning(f"Failed or timed out processing from DB: {pdf_path_in_db}")
            logging.info(
                f"DB processing summary: Successfully processed {success_count}/{total_count if total_count > 0 else 'unknown total'} PDFs.")

    except sqlite3.Error as e:
        logging.error(f"SQLite error accessing {db_path}: {e}", exc_info=True)
    return processed_papers


def load_from_folder(folder_path_str: str) -> List[Dict[str, Any]]:
    folder_path = Path(folder_path_str)
    if not folder_path.exists() or not folder_path.is_dir():
        logging.error(f"Folder not found or is not a directory: {folder_path}")
        return []

    ref_filter = ReferenceFilter(CONFIG["REF_FILTER"])
    keyword_extractor = KeywordExtractor(CONFIG["KEYWORD_EXTRACTOR"])
    processed_papers: List[Dict[str, Any]] = []

    pdf_file_paths = list(folder_path.rglob("*.pdf")) + list(folder_path.rglob("*.PDF"))
    pdf_file_paths = sorted(list(set(pdf_file_paths)))  # Deduplicate and sort

    if not pdf_file_paths:
        logging.info(f"No PDF files found in folder: {folder_path}")
        return []

    total_count = len(pdf_file_paths)
    success_count = 0

    for pdf_path in tqdm(pdf_file_paths, desc="Processing PDFs from folder", total=total_count, unit="pdf"):
        logging.info(f"Attempting to process from folder: {pdf_path}")
        pdf_data_tuple: Tuple[str] = (str(pdf_path),)  # Pass as a tuple

        result = run_with_timeout(
            process_pdf_entry,
            args_tuple=(pdf_data_tuple, ref_filter, keyword_extractor)  # Ensure args_tuple contains the tuple
        )
        if result:
            processed_papers.append(result)
            success_count += 1
            logging.info(f"Successfully processed from folder: {pdf_path.name}")
        else:
            logging.warning(f"Failed or timed out processing from folder: {pdf_path.name}")

    logging.info(f"Folder processing summary: Successfully processed {success_count}/{total_count} PDFs.")
    return processed_papers


# --- save_paper_list (largely similar) ---
def save_paper_list(papers: List[Dict[str, Any]], out_path_str: str):
    out_path = Path(out_path_str)
    try:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(papers, f, ensure_ascii=False, indent=2)
        logging.info(f"Successfully saved {len(papers)} processed paper entries to {out_path}")
    except IOError as e:
        logging.error(f"Failed to save processed papers to {out_path}: {e}", exc_info=True)


# --- Main Function (Adjusted) ---
def main():
    # --- Configuration for direct PyCharm run ---
    # Set IDE_RUN to True to use the default paths below.
    # Set IDE_RUN to False to use command-line arguments.
    IDE_RUN = True

    # --- Default values for PyCharm execution (Modify these as needed) ---
    # These are only used if IDE_RUN is True
    IDE_DEFAULT_SOURCE_TYPE = "folder"  # Options: "folder" or "db"
    IDE_DEFAULT_FOLDER_PATH = "data/raw_data"  # Path to your PDF folder
    IDE_DEFAULT_DB_PATH = "your_database.db"  # Path to your SQLite DB
    IDE_DEFAULT_OUTPUT_PATH = "data/processed_text/processed_papers.json"
    IDE_DEFAULT_EMAIL = "chenlintao3998@gmail.com"  # IMPORTANT: Set your email

    # --- Variables to hold determined paths and settings ---
    run_mode_folder_path: Optional[str] = None
    run_mode_db_path: Optional[str] = None
    run_mode_output_path: str
    run_mode_email: Optional[str] = None

    if IDE_RUN:
        logging.info("Running in IDE_RUN mode with hardcoded defaults.")
        if IDE_DEFAULT_SOURCE_TYPE == "folder":
            run_mode_folder_path = IDE_DEFAULT_FOLDER_PATH
        elif IDE_DEFAULT_SOURCE_TYPE == "db":
            run_mode_db_path = IDE_DEFAULT_DB_PATH
        else:
            logging.error(f"Invalid IDE_DEFAULT_SOURCE_TYPE: '{IDE_DEFAULT_SOURCE_TYPE}'. Choose 'folder' or 'db'.")
            return

        run_mode_output_path = IDE_DEFAULT_OUTPUT_PATH
        run_mode_email = IDE_DEFAULT_EMAIL

        if not run_mode_folder_path and not run_mode_db_path:
            logging.error("For IDE_RUN, please set a valid default input path (folder or DB) and source type.")
            return
        if not run_mode_output_path:
            logging.error("For IDE_RUN, please set a default output path.")
            return

    else:  # Use argparse for terminal runs
        logging.info("Running with command-line arguments.")
        parser = argparse.ArgumentParser(description="Advanced PDF to JSON Text Extractor")
        parser.add_argument("--folder", help="Path to folder containing PDF files.")
        parser.add_argument("--db", help="Path to SQLite database file.")
        # Made output optional for IDE runs, but it's effectively required by one of the branches
        parser.add_argument("--output", help="Path to output JSON file for results.")
        parser.add_argument("--email", help="Your email for CrossRef API (politeness). Overrides config if provided.")
        args = parser.parse_args()

        run_mode_folder_path = args.folder
        run_mode_db_path = args.db
        run_mode_output_path = args.output
        run_mode_email = args.email

        if not run_mode_folder_path and not run_mode_db_path:
            parser.error("Either --folder or --db must be specified when running from command line.")
        if not run_mode_output_path:
            parser.error("--output path must be specified when running from command line.")

    # --- Configure CrossRef Email (common logic) ---
    if run_mode_email:
        CONFIG["CROSSREF_API"]["MAILTO"] = run_mode_email
        logging.info(f"Using email for CrossRef API: {run_mode_email}")
    elif CONFIG["CROSSREF_API"]["MAILTO"] == "your_email@example.com":
        logging.warning("Using default placeholder email for CrossRef API. "
                        "Please configure CONFIG['CROSSREF_API']['MAILTO'] or use --email argument / set IDE_DEFAULT_EMAIL for polite API usage.")

    # --- Main Processing Logic (common logic) ---
    all_processed_papers: List[Dict[str, Any]] = []

    if run_mode_db_path:
        logging.info(f"Processing PDFs from database: {run_mode_db_path}")
        db_papers = load_from_db(run_mode_db_path)  # Ensure load_from_db is defined
        all_processed_papers.extend(db_papers)

    # Process folder if specified, even if DB was also processed (user might want both)
    if run_mode_folder_path:
        logging.info(f"Processing PDFs from folder: {run_mode_folder_path}")
        folder_papers = load_from_folder(run_mode_folder_path)  # Ensure load_from_folder is defined
        all_processed_papers.extend(folder_papers)
        # Optional: Add deduplication logic here if papers from DB and folder might overlap
        # For example, based on DOI or a hash of content if DOIs are not always present.

    if all_processed_papers:
        # Simple deduplication based on 'doi' if present, or 'local_path' for local files
        seen_identifiers = set()
        unique_papers = []
        for paper in all_processed_papers:
            identifier = paper.get("doi", paper.get("local_path"))
            if identifier not in seen_identifiers:
                unique_papers.append(paper)
                seen_identifiers.add(identifier)

        logging.info(f"Total papers processed (pre-deduplication): {len(all_processed_papers)}")
        logging.info(f"Total unique papers to be saved: {len(unique_papers)}")
        save_paper_list(unique_papers, run_mode_output_path)  # Ensure save_paper_list is defined
    else:
        logging.info("No papers were processed from the specified sources.")

    logging.info("PDF processing pipeline finished.")


if __name__ == "__main__":
    multiprocessing.freeze_support()  # Important for multiprocessing, especially if frozen
    main()