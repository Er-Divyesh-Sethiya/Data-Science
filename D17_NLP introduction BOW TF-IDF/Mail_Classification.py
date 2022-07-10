import pandas as pd
import re
import nltk
nltk.download('stopwords')
nltk.download('wordnet')
from nltk.stem.porter import PorterStemmer
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer,TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score,confusion_matrix
naive_bayes_model = MultinomialNB()
cv = CountVectorizer(max_features = 2500)
le = LabelEncoder()
ps = PorterStemmer()
lm = WordNetLemmatizer()


spam_ham_data = pd.read_csv('SMSSpamCollection',sep = '\t',
                            names = ['label','message'])

cleaned_data = []

for i in range(0,5572):
     review = re.sub('[^a-zA-Z]', ' ', spam_ham_data['message'][i])
     review = review.lower()
     review = review.split()
     review = [lm.lemmatize(word) for word in review if word not in stopwords.words('english')]
     review = ' '.join(review)
     cleaned_data.append(review)

X = cv.fit_transform(cleaned_data).toarray()
y = le.fit_transform(spam_ham_data['label'])

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size = 0.20,random_state = 12,stratify = y)

naive_bayes_model.fit(X_train,y_train)
y_pred = naive_bayes_model.predict(X_test)

accuracy_score(y_test, y_pred)
confusion_matrix(y_test, y_pred)
