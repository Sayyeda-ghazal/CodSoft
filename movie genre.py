import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
import matplotlib.pyplot as plt
import nltk
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords

nltk.download('stopwords')
nltk.download('wordnet')
stop_words = set(stopwords.words('english'))


train_data = pd.read_csv('/content/drive/MyDrive/ghazal/programming/Artificial Intelligence/machine learning/CodSoft/Movie Genre Classification/Genre Classification Dataset/train_data.txt', delimiter=':::')
train_data.columns = ["id", "title", "genre", "description"]
lemmatizer = WordNetLemmatizer()
train_data['description'] = train_data['description'].apply(lambda x: ' '.join([lemmatizer.lemmatize(word) for word in x.split() if word not in stop_words]))


X = train_data['description']
y = train_data['genre']


vectorizer = TfidfVectorizer()

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

# Training model
model = LogisticRegression()
model.fit(X_train_tfidf, y_train)

# Making predictions
y_pred = model.predict(X_test_tfidf)

# Calculating accuracy score
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy Score:", accuracy)

# Plot a histogram
plt.hist(y_pred, bins=len(train_data['genre'].unique()))
plt.show()


