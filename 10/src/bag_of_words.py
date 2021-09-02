import numpy as np
from sklearn.datasets import load_files
from sklearn.feature_extraction.text import CountVectorizer     # 어휘사전 구축
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS  # 불용어 목록
from sklearn.linear_model import LogisticRegression             # 분류

# from sklearn.model_selection import cross_val_score             # 교차 검증 : 실습


######################
#       전처리        #
######################

# 텍스트와 레이블을 포함하는 Bunch 오브젝트 반환
reviews_train = load_files("aclImdb_v1/aclImdb/train/")
reviews_test = load_files("aclImdb_v1/aclImdb/test/")

# Bunch 오브젝트에서 텍스트, 레이블 각각 저장
text_train, y_train = reviews_train.data, reviews_train.target
text_test, y_test = reviews_test.data, reviews_test.target

# 데이터 전처리 : HTML 태그 삭제
text_train = [doc.replace(b"<br />", b" ") for doc in text_train]
text_test = [doc.replace(b"<br />", b" ") for doc in text_test]

print("훈련 데이터의 문서 수 : ", len(text_train))
print("테스트 데이터의 문서 수  : ", len(text_test))
print("클래스별 샘플 수 (훈련 데이터) : ", np.bincount(y_train))
print("클래스별 샘플 수 (테스트 데이터) : ", np.bincount(y_test))
print()





######################
#  사전 구성, BOW 표현  #
######################

# train, test 를 토큰화, 어휘사전 구축 후 vocabulary_ 속성에 저장 (전체 단어 목록 구성)
# CountVectorizer : 빈도수 5 이상, 불용어 제거
vect = CountVectorizer(min_df=5, stop_words=ENGLISH_STOP_WORDS).fit(text_train, text_test)

# X_TRAIN, X_TEST : 희소행렬 표현
X_train = vect.transform(text_train)
X_test = vect.transform(text_test)


print("TRAIN, TEST 의 총 특징 개수 (어휘사전의 크기) : ", len(vect.get_feature_names()))
print("X_TRAIN : \n", repr(X_train))
print("X_TEST : \n", repr(X_test))
print()




######################
#    학습 및 분류      #
######################
'''
 교차검증 실습 (train만 사용)
scores = cross_val_score(LogisticRegression(max_iter=1000), X_train, y_train, cv=5)
print("교차 검증 평균 점수 : {:.4f}".format(np.mean(scores)))
print()
'''

# 학습
logreg = LogisticRegression(max_iter=1000).fit(X_train, y_train)

# 결과
print("훈련 세트 점수 : {:.3f}".format(logreg.score(X_train, y_train)))
print("테스트 세트 점수 : {:.3f}".format(logreg.score(X_test, y_test)))
print()











