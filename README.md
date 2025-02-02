# Benign Overfitting in Single-Head Attention

[0] _Benign Overfitting in Single-Head Attention_, Roey Magen et al., arXiv:2410.07746v1 [cs.LG] 10 Oct 2024

## Description
This repertory contains three Jupyter Notebook files: $overfitting_example$, $benign_overfitting$ and $different_SNR$. 
In $overfitting_example$, you will find a short experiment to vizualize what benign overfitting looks like on a very simple 2-dimensional example. 
In $benign_overfitting$, you can run two experiments from the paper [0], one using Gradient Descent and the other a Max-Margin criteria.
Lastly in $different_SNR$, we compare the impact of changing the dimension - which directly impacts the SNR.

## Models
The model is a simplified Single-Head Attention, with the assumption that the query vector is fixed, so we can rewrite the model as $f(\boldsymbol{X}; \boldsymbol{p}, \boldsymbol{v} = \boldsymbol{v}^\top \boldsymbol{X}^\top\mathcal{S}(\boldsymbol{Xp}$, where we train$\boldsymbol{v}$ and $\boldsymbol{p}$.

## Utils
- generate_data: generate a dataset using the method presented in [0], depending on different parameters
- train_model: contains the functions to train the model using different methods and to plot the results (accuracy and sample probabilities)
