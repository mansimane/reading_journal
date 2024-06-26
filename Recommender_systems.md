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
  7. Experiments: The authors did simulations in RecoGym environment and presented the results. The the methods are not deployed in real world production system. 
  8. Summary: Authoers propose to jointly optimize two bandits during training one to predict reward and another to predict action given context. The reward bandit will be used to generate training sample for action bandit. Howerver they do not explain why we can't do it one after the other. We can first optimize the reward bandit and then use it to train action bandit. Accuracy of reward bandit is very imporatant for good action bandit performance. It could be challenging to learn the reward bandit first. 


  ### On the Value of Bandit Feedback for Offline Recommender System Evaluation
  [arxiv](https://arxiv.org/abs/1907.12384)
  - Paper shows how bandit feedback would be used for effective offline evaluation that more accurately reflects online performance
  - The aurthers do k fold validation for 6 methods to predict performance. And then they do counter factual estimation using Clipped IPS for the same methods. For random sampling method, both counter factual and k fold gives bad results but it acutally does better in AB tests, this is because of bias in eval data. But for other methods, counterfactual estimate give good estimates compared to k fold offline evaluation. 
  
![Screenshot 2024-04-04 at 11 51 58 AM](https://github.com/mansimane/reading_journal/assets/23171195/6dd171a4-8376-4f3d-aae1-341737acb4c3)


### Multi-task Learning for Related Products Recommendations at Pinterest
[link](https://medium.com/pinterest-engineering/multi-task-learning-for-related-products-recommendations-at-pinterest-62684f631c12)
- ![image](https://github.com/mansimane/reading_journal/assets/23171195/b7d2b2c1-f998-4893-9a59-2981e1f80627)
- importance weight in the loss function is chosen for each engagement type according to their business values.
- where ImportanceWeight(i)is determined by the engagement type of sample i, y ⁽ᶦ⁾ binary is the true binary label, which is 1 for an impression with engagement (close-up, save, click, or long-click) or 0 otherwise, and ŷ ⁽ᶦ⁾ binary is the predicted score for sample i. This method has following drawbacks:


   1.  We lose information by combining different engagement types into one binary label. If a user both saves and long clicks a Pin, we have to drop one of the user’s actions since only one type of engagement can be chosen for each sample. An alternative would be to duplicate the training data using a different engagement per sample.
    2. The predicted score is not interpretable. It tells us how “engaging” a candidate is but its exact meaning is determined by the importance weights we choose.
    3. The task of engagement prediction is coupled with business value. If we ever want to try a different set of importance weights, we need to retrain the model, which is detrimental to developer and experimentation velocity.


  The key difference between the loss function in Eq.2 and the one in Eq.1 is that we do not lose engagement information. The four output heads can borrow knowledge from each other by sharing the previous layers, and this would also alleviate the overfitting problem compared to fitting one model for each engagement type
  but the gain is subtle because the **four tasks are similar to each other**. herefore, we ended up using equal weights for the losses for simplicity.
- Authors also experiments with Baysian approach to optimize weights, but it didn't do well in AB.

### Simulated Spotify Listening Experiences for Reinforcement Learning with TensorFlow and TF-Agents
[link](https://blog.tensorflow.org/2023/10/simulated-spotify-listening-experiences-reinforcement-learning-tensorflow-tf-agents.html)
 - Spotify leverages TensorFlow and its extended ecosystem (including TFX and TensorFlow Serving) in their production machine learning stack.
They chose TensorFlow Agents (TF-Agents) as their RL library
- Offline simulator: They designed a robust and extendable offline Spotify simulator based on TF-Agents environment primitives. This simulator allowed them to develop, train, and evaluate sequential models for item recommendations.
- Agent Models: Proximal Policy Optimization (PPG), Deep Q-Network (DQN), and a modified version called Action-Head DQN (AH-DQN).
- Live experiments demonstrated that offline performance estimates strongly correlated with online results.

- _episode_sampler - Defines environment
- _user_model - provides feedback to RL agent
- action_spec- can't be discrete as there are too many songs, _track_sampler - consumes possible recommendations and then emits actual recommendations
- observation_spec -
Terminiationand reset= Authors synthetically generate user behavior based on 

As a hypothetical, we may determine that 92% of listening sessions terminate after 6 sequential track skips and we’d construct our simulation termination logic to match, we design abstractions in our simulator that allow us to 



###  GenSERP: Large Language Models for Whole Page Presentation
  [arxiv](https://arxiv.org/pdf/2402.14301.pdf)

- Authors use Gen AI based whole page optimization for Bing.com
- There are three stages:
  - Information gathering: LLM calls diffrent APIs to extract items relevant to queries e.g. news articles related to query, ads related to query etc
  - Candidate Presentation Generation: Untile LLMs is confident, it tried diffrent UX configurations for showing the items in different rankings. The confidence score is threshould. The proposals are then rendered and given to vision model. 
  - Scoring: Candiate scores are generated based on by LLM with vision to rank quality of page visually. 
  - ![image](https://github.com/mansimane/reading_journal/assets/23171195/38044b8f-bd7f-47bf-80df-f0ac72f9753d)

- They didn't get as good results as existing bing logic, especially they did pretth bad on ecommerce, but did pretty well on healthe and entertainment.


### An Efficient Bandit Algorithm for Realtime Multivariate Optimization
 [arxiv](https://assets.amazon.science/be/1f/27be73114b198fe14869e75b6ef6/an-efficient-bandit-algorithm-for-realtime-multivariate-optimization.pdf)
 - D Slots for the page which are bandit arms, R reward, X context, A action, B_(A,X) - feature vector combining features like image selected in past, users age, time of the day etc. mu is the weight matrix,
 - 
  
### Beyond the binge: Recommending for long-term member satisfaction at Netflix
 [pdf]([https://assets.amazon.science/be/1f/27be73114b198fe14869e75b6ef6/an-efficient-bandit-algorithm-for-realtime-multivariate-optimization.pdf](https://acrobat.adobe.com/id/urn:aaid:sc:US:6421444c-6fa3-4d9a-875d-623c690986cf?viewer%21megaVerb=group-discover)
 - Long term retention is noisy and is dependent on external factors so hard to attribute
 - Instead we can train on proxy rewards such as completing the show, binge watching the show
 - But even proxy rewards can be delayed which can hurt model performance
 - As a solution policy network is trained to predict delayed reward for each example during training
 - During online setting computer proxy reward as predicted reward (for delayed feedback heads) + observable feedback
 - proxy reward = g(observed reward)
 - Challenges- proxy rewards are not aligned with online metrics

### Reward innovation for long-term member satisfaction

https://acrobat.adobe.com/id/urn:aaid:sc:US:2334c55e-5792-4864-9699-d194cfe15729?comment_id=cce62d81-e963-4fe2-b3d0-3b4e2a4dda3d
