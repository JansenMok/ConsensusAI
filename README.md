# ConsensusAI
Cross-model consensus engine for large language models (LLMs).
## Project Idea
ConsensusAI is a solo research project that aims to give the user the best possible answer by comparison of the most popular available AI LLM models available to the public.
The consensus engine queries multiple commercially available LLMs, normalizes their responses, then analyzes semantic agreement or disagreement.
- If the models broadly agree (tunable), ConsensusAI produces a concise summary of the combined responses with absolute minimal bias.
- If the models disagree, ConsensusAI highlights where the responses diverge and whether some models agree while others do not.
## Features
- **Batch Normalization**: clean noisy outputs like boilerplate, markdown, emojis, AI disclaimers, and more.
- **Agreement Engine**: compute pairwise cosine similarity, cluster responses, and detect consensus.
- **Extractive consensus**: assemble a bias-minimized answer with overlapping content within responses.
- **Testing Suite**: pytest cases for normalization and agreement detection.
### TODO
- [ ] develop output with minimal bias
- [ ] write tests for output
- [ ] develop API querying for all LLMs
