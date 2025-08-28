from pathlib import Path
from agent_compare.normalize import normalize, batch_normalize

def test_normalize_files():
    here = Path(__file__).parent
    grok = (here / "LLM_outputs" / "grok.txt").read_text(encoding="utf-8")
    claude = (here / "LLM_outputs" / "claude.txt").read_text(encoding="utf-8")

    texts = [grok, claude]
    normalized = batch_normalize(texts)

    print(normalized)
