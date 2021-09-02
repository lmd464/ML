import numpy as np
import chardet
from sklearn.datasets import load_files
from sklearn.feature_extraction.text import TfidfVectorizer     # 어휘사전 구축
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS  # 불용어 목록
from sklearn.svm import LinearSVC   # 분류
from sklearn.model_selection import cross_val_score     # 교차 검증


######################
#       전처리        #
######################

# load_files : 텍스트와 레이블을 포함하는 Bunch 오브젝트 반환
# 이후 Bunch 오브젝트에서 텍스트, 레이블 각각 저장
files = load_files("bbcsport-fulltext/bbcsport/")
X, y = files.data, files.target

# 데이터 전처리 : 인코딩 방식 변경
for i in range(len(X)):
    if(chardet.detect(X[i]) != "utf-8"):
        X[i] = X[i].decode(chardet.detect(X[i])['encoding']).encode('utf8')

# 데이터 전처리 : HTML 태그 삭제, 개행 문자 대체
X = [doc.replace(b"<br />", b"") for doc in X]
X = [doc.replace(b"\n", b" ") for doc in X]

print("훈련 데이터의 문서 수 : ", len(X))
print("클래스별 샘플 수 : ", np.bincount(y))
print()



######################
#  사전 구성, BOW 표현  #
######################

# 어휘사전 구축 후 vocabulary_ 속성에 저장 (전체 단어 목록 구성)
# TfifdVectorizer : 불용어 제거, L2 norm
vect = TfidfVectorizer(input=X, stop_words=ENGLISH_STOP_WORDS, norm='l2').fit(X)

# X : 희소행렬 표현
X_mat = vect.transform(X)



######################
#    학습 및 분류      #
######################

# 학습 및 교차검증 (fold : 5)
scores = cross_val_score(LinearSVC(max_iter=10000000), X_mat, y, cv=5)
print("교차 검증 평균 점수 : {:.4f}".format(np.mean(scores)))
print()
