
# Machine Learning-Powered Wine Classifier

This project uses the Nearest Centroid Classifier model to classify wines as either red or white wines. Uses k-fold cross validation to increase accuracy of classification.

---

## Dataset

The dataset was obtained from P.Cortez, A. Cerdeira, F. Almeida, T. Matos, and J. Reiss in their original study titled "Modeling wine preferences by data mining from physicochemical properties". The datasets mentioned in the paper are repurposed in my project to classify wines into red or white wines, rather than predicting preferences. 

The data is comprised of two CSV files, one with 1599 red wines and the other with 1599 white wines. Each wine is described as a point in a 12-dimensional space. For the sake of this project, we ignore the "quality" description and treat each wine as a point in a 11-dimensional space.

---

## Training mechanism

The program is trained on the set of 3198 data points by computing the centroid of all white wines and the centroid of all red wines using the ```compute_centroid``` function. Test wines are then compared to the red wine centroid and the white wine centroid. The program predicts whether an unknown wine is red or white by comparing the euclidean distance between the unknown wine and the two centroids. 

To maximize prediction accuracy and increase training count, I opted to use a k-fold cross validation model to allow for my limted dataset to be used more efficiently. 

---

## Results

k-fold cross validation produced the following accuracies:

- k = 4: 0.882
- k = 6: 0.883
- k = 8: 0.884
- k = 10: 0.881

Here is the result of the 8-fold cross validation experiment:
```
Number of predictions: 400
Number of correct predictions: 333
Accuracy: 0.8325
Number of predictions: 400
Number of correct predictions: 354
Accuracy: 0.885
Number of predictions: 400
Number of correct predictions: 353
Accuracy: 0.8825
Number of predictions: 400
Number of correct predictions: 353
Accuracy: 0.8825
Number of predictions: 400
Number of correct predictions: 356
Accuracy: 0.89
Number of predictions: 400
Number of correct predictions: 361
Accuracy: 0.9025
Number of predictions: 400
Number of correct predictions: 361
Accuracy: 0.9025
Number of predictions: 398
Number of correct predictions: 352
Accuracy: 0.8844221105527639
8-fold cross-validation accuracy: 0.8827402638190954
```