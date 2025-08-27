# testcase
with open("test.txt", "r") as file:
    test = file.read()

import re
from typing import Iterable

###      regex definitions      ###
_CODE_FENCE = re.compile(r"^```(?:[\w.+-]+)?\n(.*?)\n```$", re.S)
_MARKDOWN_LINK = re.compile(r"\[([^\]]+)\]\(([^)]+)\)")
_FRONTMATTER = re.compile(r"^---\n.*?\n---\n", re.S)
_AI_PREFACE = re.compile(
    r"^(?:as\s+an\s+ai(?:\s+language\s+model)?[:,\s].*?[\.\!]\s*)",  # test
    re.I | re.S
)
_MULTI_WS = re.compile(r"[ \t]{2,}")
_MULTI_NL = re.compile(r"\n{3,}")

# to catch emoji headers
_HEADING_WORDS = r"(?:tl;?\s*dr|summary|overview|conclusion|key\s+takeaways?|key\s+points?|highlights?)"
_HEADING_ONLY_LINE = re.compile(
    rf"(?mi)^[\s\W]*{_HEADING_WORDS}\s*(?:[:\-–—•]*)\s*$"
)

# catches trailing follow-up offers
_TRAILING_FOLLOWUP = re.compile(
    r"(?i)(?:\s*(?:would you like|shall i|should i|do you want|can i|let me know).*?\?\s*)+$"
)

###      helpers      ###
def _strip_code_fences(text: str) -> str:
    m = _CODE_FENCE.match(text.strip())
    return m.group(1) if m else text

def _strip_md_links(text: str) -> str:
    # Convert [label](url) -> label
    return _MARKDOWN_LINK.sub(r"\1", text)

# YAML metadata
def _strip_front_matter(text: str) -> str:
    return _FRONTMATTER.sub("", text, count=1)

def _strip_ai_preface(text: str) -> str:
    return _AI_PREFACE.sub("", text)

# def _strip_lists_boilerplate(text: str) -> str:
#     # Remove repeated headers like "Here’s a summary:" that add noise
#     text = re.sub(r"^\s*(?:summary|overview|conclusion)[:\-]\s*", "", text, flags=re.I)
#     return text
def _strip_heading_only_lines(text: str) -> str:
    # remove decorative emojis
    return _HEADING_ONLY_LINE.sub("", text)

def _strip_trailing_followup(text: str) -> str:
    return _TRAILING_FOLLOWUP.sub("", text).strip()

# cleanup
def normalize(text: str) -> str:
    t = text or ""
    t = _strip_front_matter(t)
    t = _strip_code_fences(t)
    t = _strip_md_links(t)
    t = _strip_ai_preface(t)
    t = _strip_heading_only_lines(t)   # <-- new
    t = t.replace("\r\n", "\n").replace("\r", "\n")
    t = _MULTI_WS.sub(" ", t)
    t = _MULTI_NL.sub("\n\n", t)
    t = _strip_trailing_followup(t)
    return t.strip()

def batch_normalize(texts: Iterable[str]) -> list[str]:
    return [normalize(t) for t in texts]

print(normalize(test))
