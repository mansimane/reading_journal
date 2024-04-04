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


###  Learning from Bandit Feedback: AnOverview of the State-of-the-art
[arxiv](https://arxiv.org/abs/1909.08471)
[code](https://github.com/leoguelman/BLBF)
- It is diffficult to estimate peroformance of policies that frequently perform actions that were infrequently done in the past.
- Through importance sampling and variance reduction techniques, CRM (Counter factual risk minimization) methods allow robust learning.
- Notations
  1. User state x R_n, action a, reward c
  2. Train a logistic regression model to predit (C|x,a). But this model w![Screenshot 2024-04-04 at 10 27 55 AM](https://github.com/mansimane/reading_journal/assets/23171195/61f49f5a-23bd-4dd6-bff1-bceb5436d2a3)
ill underfit to examples where there is less data x,a pairs if a was not recommendeded often in the past. So during the training, the samples (x_i, a_i, c_i) are re-weighted using inverse propensity score. (NOTE: this is beneficial only if model lacks capacity to correctly model complete relationship).
![Uploading Screenshot 2024-04-04 at 10.27.55 AM.png…]()

  4. Contexual bandits: to predict probability of the action given the context P(a|x), 
  5. The authors propouse to train a contextual bandit to predict P(a|x) and another bandit to predict  P(c|a,x) jointly. 
  6. Dual bandit: LLdual(θ) = (1 −α)LCB(θ) + αLLH(θ)
  7. Summary: Authoers propose to jointly optimize two bandits during training one to predict reward and another to predict action given context. The reward bandit will be used to generate training sample for action bandit. Howerver they do not explain why we can't do it one after the other. We can first optimize the reward bandit and then use it to train action bandit. 


  ### On the Value of Bandit Feedback for Offline Recommender System Evaluation
  [arxiv](https://arxiv.org/abs/1907.12384)
  - Paper shows how bandit feedback would be used for effective offline evaluation that more accurately reflects online performance
  - The aurthers do k fold validation for 6 methods to predict performance. And then they do counter factual estimation using Clipped IPS for the same methods. For random sampling method, both counter factual and k fold gives bad results but it acutally does better in AB tests. But for other methods, counterfactual estimate give good estimates compared to k fold offline evaluation. 
  
![Screenshot 2024-04-04 at 11 51 58 AM](https://github.com/mansimane/reading_journal/assets/23171195/6dd171a4-8376-4f3d-aae1-341737acb4c3)
