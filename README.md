# Scientific-Impact-Aligned-Pairwise-Comparative-Evaluation-LLM
We propose a two-step training strategy using reinforcement learning to train a Scientific Impact-Aligned Pairwise Comparative Evaluation LLM, which can generate the relative quality relationship and scores that are well-aligned with papers’ scientific impact. 


(1). The labeled dataset includs the trainng, validation, and test set, which are used to contructed and evaluate our model. The PMIDs of papers are provided in these excel.csv. 
(2). The expert-recommended and prize-winding datasets include the metadata of the expeirmental and control groups. Additionally, we aslo provide the comparative papers for the experimental and control groups, which are retrieved from our self-constructed vector database.
(3). The two random datasets (papers published in 2010 and 2024) also include the PMID, journal name, citation count, and etc.
