import pandas as pd
import numpy as np
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
import re
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, roc_curve, auc, accuracy_score

# Veri setini yükle
yorumlar = pd.read_csv('Restaurant_Reviews.csv')  # Dosya yolunu doğru şekilde girin

# Stop word'leri indir
nltk.download('stopwords')

# Stemmer ve stop word listesini oluştur
ps = PorterStemmer()
stop_words = set(stopwords.words('english'))

# Yorumları işle
derlem = []
for i in range(len(yorumlar)):
    # Eğer veri eksikse, atla
    if pd.isnull(yorumlar['Review'][i]):
        continue
    yorum = yorumlar['Review'][i]
    
    # Yorumları sadece harfler içerecek şekilde temizle
    yorum = re.sub('[^a-zA-Z]', ' ', yorum)
    yorum = yorum.lower().split()  # Küçük harfe çevir ve kelimelere ayır
    
    # Stopword'leri çıkar ve stemming uygula
    yorum = [ps.stem(kelime) for kelime in yorum if not kelime in stop_words]
    
    # Temizlenmiş kelimeleri tekrar birleştir
    yorum = ' '.join(yorum)
    derlem.append(yorum)
    
# Yorumları sayısal verilere dönüştür
from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer(max_features=1000)
X = cv.fit_transform(derlem).toarray()

# Hedef değişken (label) y'yi oluştur
y = yorumlar.iloc[:, 1].values

# Eksik verileri çıkar
X = X[~pd.isnull(y)]
y = y[~pd.isnull(y)]

# Veriyi eğitim ve test setlerine ayır
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=0)

# Confusion matrix ve doğruluğu yazdırma fonksiyonu
def confusion_matrix_print(y_test, y_pred, classifier_name):
    cm = confusion_matrix(y_test, y_pred)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Confusion Matrix for {classifier_name}:\n", cm)
    print(f"{classifier_name} - Accuracy: {accuracy:.2f}\n")

# ROC ve AUC değerlerini yazdırma fonksiyonu
def roc_metrics(y_test, y_proba, classifier_name):
    fpr, tpr, _ = roc_curve(y_test, y_proba)
    roc_auc = auc(fpr, tpr)
    
    # TPR ve FPR hesaplanıyor
    print(f"{classifier_name} - True Positive Rate (TPR): {tpr[1]:.2f}")
    print(f"{classifier_name} - False Positive Rate (FPR): {fpr[1]:.2f}")
    print(f"{classifier_name} - AUC: {roc_auc:.2f}\n")

# 1. Gaussian Naive Bayes
from sklearn.naive_bayes import GaussianNB
classifier_nb = GaussianNB()
classifier_nb.fit(X_train, y_train)
y_pred_nb = classifier_nb.predict(X_test)
y_proba_nb = classifier_nb.predict_proba(X_test)[:, 1]  # Pozitif sınıf olasılıkları
confusion_matrix_print(y_test, y_pred_nb, "Gaussian Naive Bayes")
roc_metrics(y_test, y_proba_nb, "Gaussian Naive Bayes")

# 2. Logistic Regression
from sklearn.linear_model import LogisticRegression
classifier_lr = LogisticRegression(max_iter=1000)
classifier_lr.fit(X_train, y_train)
y_pred_lr = classifier_lr.predict(X_test)
y_proba_lr = classifier_lr.predict_proba(X_test)[:, 1]
confusion_matrix_print(y_test, y_pred_lr, "Logistic Regression")
roc_metrics(y_test, y_proba_lr, "Logistic Regression")

# 3. Support Vector Machine (SVM)
from sklearn.svm import SVC
classifier_svm = SVC(kernel='linear', probability=True, random_state=0)  # probability=True olasılık tahminini açar
classifier_svm.fit(X_train, y_train)
y_pred_svm = classifier_svm.predict(X_test)
y_proba_svm = classifier_svm.predict_proba(X_test)[:, 1]
confusion_matrix_print(y_test, y_pred_svm, "Support Vector Machine")
roc_metrics(y_test, y_proba_svm, "Support Vector Machine")

# 4. Random Forest Classifier
from sklearn.ensemble import RandomForestClassifier
classifier_rf = RandomForestClassifier(n_estimators=100, random_state=0)
classifier_rf.fit(X_train, y_train)
y_pred_rf = classifier_rf.predict(X_test)
y_proba_rf = classifier_rf.predict_proba(X_test)[:, 1]
confusion_matrix_print(y_test, y_pred_rf, "Random Forest")
roc_metrics(y_test, y_proba_rf, "Random Forest")

# 5. K-Nearest Neighbors (KNN)
from sklearn.neighbors import KNeighborsClassifier
classifier_knn = KNeighborsClassifier(n_neighbors=5)
classifier_knn.fit(X_train, y_train)
y_pred_knn = classifier_knn.predict(X_test)
y_proba_knn = classifier_knn.predict_proba(X_test)[:, 1]
confusion_matrix_print(y_test, y_pred_knn, "K-Nearest Neighbors")
roc_metrics(y_test, y_proba_knn, "K-Nearest Neighbors")

# 6. Decision Tree Classifier
from sklearn.tree import DecisionTreeClassifier
classifier_dt = DecisionTreeClassifier(random_state=0)
classifier_dt.fit(X_train, y_train)
y_pred_dt = classifier_dt.predict(X_test)
y_proba_dt = classifier_dt.predict_proba(X_test)[:, 1]
confusion_matrix_print(y_test, y_pred_dt, "Decision Tree")
roc_metrics(y_test, y_proba_dt, "Decision Tree")
