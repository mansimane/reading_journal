### Large Language Models are Zero-Shot Rankers for Recommender Systems
[arxiv](https://arxiv.org/pdf/2305.08845.pdf#:~:text=LLMs%20outperform%20existing%20zero%2Dshot,models%20with%20different%20practical%20strategies.)
- The paper explores zero shot ranking capacity of GPT 3.5
- For prompting they use a few tricks like
  - assigning sequence number to items,
  - explicitly highlighting rececnty in sequence by saying "Notethatmy most recently watched movie is Dead Presidents. "
  - providing one example for the current user as in-context learning " If I’ve watched the following movies in the past in order: ’0. Multiplicity’, ’1. Jurassic Park’, . . .,
then you should recommend Dead Presidents to me and now that I’ve watched Dead Presidents, then ...”."

#### Challenges
  - Output of LLMs is in natural language text so they use heuristic text-matching methods and ground the recommendation results on the specified item set. This adds extra latency in production level applications
  - LLMs geenrated output which is out of candidate set 3% of the time.
  - LLms strugged to perceive order.

Observation:
  LLMs perform worse than trained models on interaction data but perform better than zero shot methods.

### TALLRec: An Effective and Efficient Tuning Framework to Align Large Language Model with Recommendation
[arxiv](https://arxiv.org/pdf/2305.00447.pdf)
[code]( https://anonymous.4open.science/r/LLM4Rec-Recsys)
- TALLRec: Tuning framework for Aligning LLMs with Recommendation, namely TALLRec
