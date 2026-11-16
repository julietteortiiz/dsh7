# HW07: Logistic Regression and Visualization

Name(s):

## Setup and tests

Use Python 3.10+. Install `python3 -m pip install -r requirements.txt`.
Run `python3 -m unittest -v test_hw07`. Complete the nine TODO functions;
read their typed signatures and docstrings. Also implement the sklearn comparison
in the `if args.compare:` block of `run_LR.py`. The flag is provided, but the
comparison code is yours to write. The tests intentionally fail before implementation.
See the course HW07 page for the complete instructions.

## Reproduce experiments

After implementing your model and sklearn comparison:

```bash
python3 run_LR.py -r data/phoneme_train.csv -e data/phoneme_test.csv -a 0.02 --compare
python3 run_LR.py -r data/challenger_train.csv -e data/challenger_test.csv -a 0.0001 --tmax 10000 --eps 0 --compare --plot challenger.pdf
```

Record learning rate, tolerance, epochs, seed, and any training-only validation.
Report accuracy and confusion matrix (rows=true, columns=predicted) for both
models on both datasets. Include Challenger's probability of damage at 31°F.

## Analysis

1. Are there any differences between your implementation of logistic
   regression and the sklearn model? How did the sklearn model perform on
   the datasets? Include the settings, accuracies, and confusion matrices.
2. So far, we have run logistic regression and Naive Bayes on different
   types of datasets. Why is that? Discuss the types of features and labels
   that we worked with in each case.
3. Regarding the Challenger data specifically, what was your final predicted
   probability of O-ring damage after fitting the model? Is this what you
   expected? Based on the available evidence, would you have recommended
   Challenger be launched on 1/28/86? What data and/or modeling might have
   helped increase confidence in the prediction? Distinguish damage from a
   catastrophic accident and discuss extrapolation.

## Feedback (ungraded)

Hours spent:
Difficulty (1–5):
Biggest challenge:

## Submission

Submit LogisticRegression.py, run_LR.py, README.md, challenger.pdf,
requirements.txt and test_hw07.py at the root of your ZIP. 
# dsh7
