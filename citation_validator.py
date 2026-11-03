
import re


SOURCE_ID_PATTERN = re.compile(
    r"(?:\[\s*(?P<ascii>S\d+)\s*\]|【\s*(?P<full_width>S\d+)\s*】)",
    re.IGNORECASE,
)


def _normalize_source_id(match):
    """Return the source ID captured from either supported bracket format."""
    return (match.group("ascii") or match.group("full_width")).upper()


def _citation_only(text):
    return re.fullmatch(
        rf"\s*(?:{SOURCE_ID_PATTERN.pattern}\s*)+",
        text,
        flags=re.IGNORECASE,
    ) is not None


def _sentence_units(text):
    citation_sequence = rf"(?:{SOURCE_ID_PATTERN.pattern}\s*)+"

    # Treat a citation after sentence punctuation as attached to that sentence.
    text = re.sub(
        rf"(?P<punctuation>[.!?])\s*(?P<citation>{citation_sequence})(?=\s|$)",
        lambda match: f" {match.group('citation')}{match.group('punctuation')}",
        text,
        flags=re.IGNORECASE,
    )

    protected = []

    def protect(value):
        protected.append(value)
        return f"\ue000{len(protected) - 1}\ue001"

    # Keep punctuation inside Markdown code and URLs from creating boundaries.
    text = re.sub(r"```[\s\S]*?```|`[^`\n]*`", lambda match: protect(match.group()), text)

    def protect_url(match):
        value = match.group()
        suffix = ""
        while value and value[-1] in ".,!?;:":
            suffix = value[-1] + suffix
            value = value[:-1]
        return protect(value) + suffix if value else suffix

    text = re.sub(r"https?://[^\s<>()]+", protect_url, text, flags=re.IGNORECASE)

    # Protect common abbreviations, initialisms, and decimal points.
    text = re.sub(
        r"\b(?:e\.g|i\.e|Mr|Mrs|Ms|Dr|Prof|etc|vs|No|Fig)\.",
        lambda match: match.group().replace(".", "\ue002"),
        text,
        flags=re.IGNORECASE,
    )
    text = re.sub(
        r"\b(?:[A-Za-z]\.){2,}",
        lambda match: match.group().replace(".", "\ue002"),
        text,
    )
    text = re.sub(r"(?<=\d)\.(?=\d)", "\ue002", text)

    def restore(unit):
        unit = unit.replace("\ue002", ".")
        return re.sub(
            r"\ue000(\d+)\ue001",
            lambda match: protected[int(match.group(1))],
            unit,
        )

    return [
        restore(unit.strip())
        for unit in re.split(r"(?<=[.!?])\s+", text)
        if unit.strip()
    ]


def validate_citation_structure(answer):
    """Require a source ID at the end of each sentence or procedural step.

    This is a structural check only. It does not determine whether a cited
    passage semantically supports the sentence.
    """
    blocks = [
        block
        for block in re.split(r"\n\s*\n", answer or "")
        if block.strip()
    ]
    uncited_claims = []
    trailing_citation_block = False
    claim_count = 0

    for block_index, block in enumerate(blocks):
        lines = [line.strip() for line in block.splitlines() if line.strip()]
        if not lines:
            continue

        if all(_citation_only(line) for line in lines):
            if block_index == len(blocks) - 1:
                trailing_citation_block = True
            continue

        paragraph_lines = []

        def check_units(text):
            nonlocal claim_count
            text = re.sub(r"^\s*(?:[-*+]\s+|\d+[.)]\s+)", "", text)
            for unit in _sentence_units(text):
                unit = unit.strip()
                if not unit or unit.startswith("#"):
                    continue
                if re.fullmatch(r"\|?\s*[:|\- ]+\s*\|?", unit):
                    continue
                claim_count += 1
                if not re.search(
                    rf"(?:{SOURCE_ID_PATTERN.pattern})\s*[.!?]*\s*$",
                    unit,
                    flags=re.IGNORECASE,
                ):
                    uncited_claims.append(unit)

        def flush_paragraph():
            if paragraph_lines:
                check_units(" ".join(paragraph_lines))
                paragraph_lines.clear()

        for line in lines:
            if re.match(r"^#{1,6}\s+", line):
                flush_paragraph()
                continue

            list_item = re.match(r"^(?:[-*+]\s+|\d+[.)]\s+)", line)
            table_row = line.startswith("|")
            if list_item or table_row:
                flush_paragraph()
                check_units(line)
            else:
                paragraph_lines.append(line)

        flush_paragraph()

    if trailing_citation_block:
        uncited_claims.append("Standalone trailing citation block")

    return {
        "is_valid": claim_count > 0 and not uncited_claims,
        "claim_count": claim_count,
        "uncited_claims": uncited_claims,
        "trailing_citation_block": trailing_citation_block,
    }


def build_source_map(retrieved_chunks):
    """Assign stable, per-passage source IDs in prompt order."""
    return {
        f"S{index}": int(chunk["page_number"])
        for index, chunk in enumerate(retrieved_chunks, start=1)
    }


def resolve_source_citations(answer, retrieved_chunks):
    """Validate model-facing source IDs and render only valid PDF page citations."""
    source_map = build_source_map(retrieved_chunks)
    source_citation_occurrences = []
    cited_source_ids = []

    for match in SOURCE_ID_PATTERN.finditer(answer):
        source_id = _normalize_source_id(match)
        source_citation_occurrences.append(source_id)
        if source_id not in cited_source_ids:
            cited_source_ids.append(source_id)

    valid_source_ids = [source_id for source_id in cited_source_ids if source_id in source_map]
    unknown_source_ids = [source_id for source_id in cited_source_ids if source_id not in source_map]

    def replace_source_id(match):
        source_id = _normalize_source_id(match)
        return f"【{source_map[source_id]}】" if source_id in source_map else match.group(0)

    return {
        "answer": SOURCE_ID_PATTERN.sub(replace_source_id, answer),
        "source_map": source_map,
        "has_citations": bool(cited_source_ids),
        "source_citation_occurrences": source_citation_occurrences,
        "cited_source_ids": cited_source_ids,
        "valid_source_ids": valid_source_ids,
        "unknown_source_ids": unknown_source_ids,
        "is_valid": bool(cited_source_ids) and not unknown_source_ids,
    }


def validate_citations(answer, retrieved_chunks):

    normalized_answer = re.sub(r"\s+", " ", answer)

    citation_pattern = (
        r"\bPages?\s+"
        r"(\d+(?:\s*(?:,|and|&)\s*\d+)*)"
        r"|【\s*(\d+)\s*】"
    )

    cited_pages = []

    for match in re.finditer(
        citation_pattern,
        normalized_answer,
        flags=re.IGNORECASE
    ):
        if match.group(1):
            numbers = re.findall(r"\d+", match.group(1))
        else:
            numbers = [match.group(2)]

        cited_pages.extend(int(number) for number in numbers)

    cited_pages = sorted(set(cited_pages))

    retrieved_pages = {
        int(chunk["page_number"])
        for chunk in retrieved_chunks
    }

    valid_pages = [
        page for page in cited_pages
        if page in retrieved_pages
    ]

    invalid_pages = [
        page for page in cited_pages
        if page not in retrieved_pages
    ]

    return {
        "has_citations": bool(cited_pages),
        "cited_pages": cited_pages,
        "retrieved_pages": sorted(retrieved_pages),
        "valid_pages": valid_pages,
        "invalid_pages": invalid_pages,
        "is_valid": bool(cited_pages) and not invalid_pages
    }
