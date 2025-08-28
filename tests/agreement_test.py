from pathlib import Path
from agent_compare.normalize import normalize, batch_normalize
from agent_compare.agreement import agreement_decision

def test_agreement():
    here = Path(__file__).parent
    grok = (here / "LLM_outputs" / "grok.txt").read_text(encoding="utf-8")
    claude = (here / "LLM_outputs" / "claude.txt").read_text(encoding="utf-8")

    texts = [grok, claude]
    normalized = batch_normalize(texts)

    print(agreement_decision(normalized))
