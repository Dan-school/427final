# 427final
## Daniel Lindberg

### Overview
The problem faced is the problem that we all face, categorizing biomedical research articles pertaining to "acute rheumatic arthritis", "disease, lyme", "abnormalities, cardiovascular", and "knee osteoarthritis". I'm going to do this with bag of words though naive Bayes, SVM, and logistical prediction models to predict the class based on the abstract.

### Libraries used
- biopython
- nltk
- sklearn
- xml
- matplotlib

### The dataset
The dataset will come from pubmed, and as per the instructions, pull all the documents that were published after 2010 that have one of the four classifications list above. Thank fully I heard about BioPython so all of the articles were pulled using that making it much simpler, and then storing them in an xml file for later use.

### Method
So like said in the dataset portion, biopython was used to collect articles. After the data was collected though, I used my normalize method from homework 2 to normalize the articles. It takes the string, tokenizes it removing punctuation, then removes stop words, lower cases the words, port stems, then lemmatizes the words, finally returning a normailzed string. After the dataset was split into a training set of 80/20.
### Validation and results
#### Naive Bayes
```
pipe  = Pipeline([
    ('vect', CountVectorizer(ngram_range=(1,2))),
    ('tfidf', TfidfTransformer( use_idf=True)),
    ('clf', MultinomialNB( alpha=0.1, fit_prior=True, class_prior=None)),
])
```
![plot](/confusion_matrix_Bayes.png)
```
Accuracy: 0.960635
Precision: 0.961
Recall: 0.961
F1: 0.961
                               precision    recall  f1-score   support

    acute rheumatic arthritis       0.94      0.96      0.95      1617
                disease, lyme       0.95      0.92      0.94      1550
abnormalities, cardiovascular       0.98      0.96      0.97      1644
          knee osteoarthritis       0.96      0.99      0.98      1870

                     accuracy                           0.96      6681
                    macro avg       0.96      0.96      0.96      6681
                 weighted avg       0.96      0.96      0.96      6681
```
#### SVM
```
pipe = Pipeline([
    ('vect', TfidfVectorizer(max_df=0.9,ngram_range=(1,2))),
    ('clf', SGDClassifier(loss='hinge', penalty='l1', alpha=1e-5, max_iter=100, random_state=42)),
])
```
![plot](/confusion_matrix_SVM.png)
```
Accuracy: 0.978147
Precision: 0.978
Recall: 0.978
F1: 0.978
                               precision    recall  f1-score   support

    acute rheumatic arthritis       0.95      0.98      0.97      1617
                disease, lyme       0.97      0.96      0.97      1550
abnormalities, cardiovascular       1.00      0.98      0.99      1644
          knee osteoarthritis       0.99      0.99      0.99      1870

                     accuracy                           0.98      6681
                    macro avg       0.98      0.98      0.98      6681
                 weighted avg       0.98      0.98      0.98      6681
```
#### Logistic Regression
```
pipe = Pipeline([   #Pipeline for logistic regression
    ('vect', TfidfVectorizer(max_df=0.8)),
    ('clf', LogisticRegression(penalty='l2', C=3, random_state=42,
                                 max_iter=1000)),
])
```
![plot](/confusion_matrix_LR.png)
```
Accuracy: 0.973208
Precision: 0.973
Recall: 0.973
F1: 0.973
                               precision    recall  f1-score   support

    acute rheumatic arthritis       0.95      0.98      0.96      1617
                disease, lyme       0.97      0.95      0.96      1550
abnormalities, cardiovascular       0.99      0.97      0.98      1644
          knee osteoarthritis       0.99      0.99      0.99      1870

                     accuracy                           0.97      6681
                    macro avg       0.97      0.97      0.97      6681
                 weighted avg       0.97      0.97      0.97      6681
```
### Conclusion
All three of the pipelines created worked exceptionally well. The lowest accuracy was the quickest performing, it was originally 93.4% but by using ngrams I was able to increase accuracy to 96%. The most accurate was the SVM prediction pipeline coming in at 97.8% accurate. Over all the results confirm that the goal set has been reached.

### Discussion
While the goal that was set was reached there are still many improvements that could be done to these pipelines. For the first one, it did pretty well on the cardiovascular portion with an 98% precision, but the other 3 didn't do so well. I tried multiple different settings for the pipeline and wasn't able to see much of a improvement, other than adding more ngrams but that multiplied the time it took to run, to the point it wasnt worth it. The SVM pipeline did really well but the cardiovascular is a litte suspicious. At 100% precision it could be over fitting for that classification, in order to test that though I would need more articles with abstracts of that classification. The same thing could be happening to the knee classification for this pipeline. For the last linear regression the same thing would be happening to the same classifications as the SVM pipeline. But there is also another issue with "acute rheumatic arthritis". The pipeline is having issues classifing 1 as 0 which is pulling down the overall accuracy. 

If I were to continue this project, I would give Word2vec a go. I didn’t get to it for this project because of other class projects and the amount of time it would take to run those. The BoW seemed accurate enough to be passable though, with the lowest accuracy 96%. I would also be interested in greating a paer of the code to categorize new unseen articles and test them on the created models to see how the prediction models work on more articles.
