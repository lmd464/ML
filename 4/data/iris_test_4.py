#!/usr/bin/env python
# coding: utf-8

# In[4]:


import numpy as np


data_input = np.loadtxt("C:\\Users\\l4m6d4\\Desktop\\3-2\\기계학습\\ML_04_201702081_최재범\\iris.csv", 
                        delimiter = ',', 
                        dtype = np.float32)


# 다른 군집 끼리 같은 인덱스를 사용하기 위해 군집별로 리스트 분리
data_1 = data_input[0:50, :]
data_2 = data_input[50:100, :]
data_3 = data_input[100:150, :]



# Cross Validation 데이터 생성
# 전체 데이터를, test (30개) 와 train (120개) 으로 쪼개기
# test 1 ~ 5 / train 1 ~ 5 : 시행 횟수마다 사용할 데이터

test_1 = np.vstack( (np.vstack( (data_1[0:10, 0:4], data_2[0:10, 0:4]) ), data_3[0:10, 0:4]) )
train_1 = np.vstack( (np.vstack( (data_1[10:50, 0:4], data_2[10:50, 0:4]) ), data_3[10:50, 0:4]) )

test_2 = np.vstack( (np.vstack( (data_1[10:20, 0:4], data_2[10:20, 0:4]) ), data_3[10:20, 0:4]) )
train_2 = np.vstack( (np.vstack( ( np.vstack((data_1[0:10, 0:4], data_1[20:50, 0:4])), np.vstack((data_2[0:10, 0:4], data_2[20:50, 0:4])) ) ), np.vstack((data_3[0:10, 0:4], data_3[20:50, 0:4])) ) )

test_3 = np.vstack( (np.vstack( (data_1[20:30, 0:4], data_2[20:30, 0:4]) ), data_3[20:30, 0:4]) )
train_3 = np.vstack( (np.vstack( ( np.vstack((data_1[0:20, 0:4], data_1[30:50, 0:4])), np.vstack((data_2[0:20, 0:4], data_2[30:50, 0:4])) ) ), np.vstack((data_3[0:20, 0:4], data_3[30:50, 0:4])) ) )

test_4 = np.vstack( (np.vstack( (data_1[30:40, 0:4], data_2[30:40, 0:4]) ), data_3[30:40, 0:4]) )
train_4 = np.vstack( (np.vstack( ( np.vstack((data_1[0:30, 0:4], data_1[40:50, 0:4])), np.vstack((data_2[0:30, 0:4], data_2[40:50, 0:4])) ) ), np.vstack((data_3[0:30, 0:4], data_3[40:50, 0:4])) ) )

test_5 = np.vstack( (np.vstack( (data_1[40:50, 0:4], data_2[40:50, 0:4]) ), data_3[40:50, 0:4]) )
train_5 = np.vstack( (np.vstack( (data_1[0:40, 0:4] , data_2[0:40, 0:4]) ), data_3[0:40, 0:4]) )


                    
# test 1 ~ 5 실험 분류 결과 : 최종 목표
exp_classify_list_test_1 = np.zeros(30)
exp_classify_list_test_2 = np.zeros(30)
exp_classify_list_test_3 = np.zeros(30)
exp_classify_list_test_4 = np.zeros(30)
exp_classify_list_test_5 = np.zeros(30)


# train 1 ~ 5 : 거리를 저장할 리스트 생성 : 1-NN 선택에 사용
dist_list_train_1 = np.zeros(120)
dist_list_train_2 = np.zeros(120)
dist_list_train_3 = np.zeros(120)
dist_list_train_4 = np.zeros(120)
dist_list_train_5 = np.zeros(120)


# test 실제 분류 결과
real_classify_list_test = np.zeros(30)
real_classify_list_test[0:10] = 1
real_classify_list_test[10:20] = 2
real_classify_list_test[20:30] = 3


# train 실제 분류 결과 : 1-NN 선택에 사용
real_classify_list_train = np.zeros(120)
real_classify_list_train[0:40] = 1
real_classify_list_train[40:80] = 2
real_classify_list_train[80:120] = 3





# 거리 계산
def dist(vector_a, vector_b):
    sub = vector_a - vector_b
    sum = np.sum(sub ** 2)
    dist = sum ** 0.5
    return dist



# 거리 기준 분류
def classify_experiment(test, train, dist_list_train, exp_classify_list_test):

    # 행렬 탐색
    for test_row in range(0, test.shape[0]):
        for train_row in range(0, train.shape[0]):
            dist_test_train = dist(test[test_row], train[train_row])
            dist_list_train[train_row] = dist_test_train
        min_dist_index = dist_list_train.argmin()
        exp_classify_list_test[test_row] = real_classify_list_train[min_dist_index]
    
    return exp_classify_list_test


classify_1 = classify_experiment(test_1, train_1, dist_list_train_1, exp_classify_list_test_1)
classify_2 = classify_experiment(test_2, train_2, dist_list_train_2, exp_classify_list_test_2)
classify_3 = classify_experiment(test_3, train_3, dist_list_train_3, exp_classify_list_test_3)
classify_4 = classify_experiment(test_4, train_4, dist_list_train_4, exp_classify_list_test_4)
classify_5 = classify_experiment(test_5, train_5, dist_list_train_5, exp_classify_list_test_5)
                    


# 실험한 군집값과 실제 군집값의 비교 후, 정확도 반환
def accurancy(exp_classify_list_test, real_classify_list_test):
    
    data_amount = len(exp_classify_list_test)
    
    correct_count = 0
    for list_index in range(0, data_amount):
        if exp_classify_list_test[list_index] == real_classify_list_test[list_index]:
            correct_count = correct_count + 1
               
    return (correct_count / data_amount) * 100



accurancy_1 = accurancy(classify_1, real_classify_list_test)
accurancy_2 = accurancy(classify_2, real_classify_list_test)
accurancy_3 = accurancy(classify_3, real_classify_list_test)
accurancy_4 = accurancy(classify_4, real_classify_list_test)
accurancy_5 = accurancy(classify_5, real_classify_list_test)
average = (accurancy_1 + accurancy_2 + accurancy_3 + accurancy_4 + accurancy_5) / 5


print("1. 정확도 : %f %%" % accurancy_1)
print("2. 정확도 : %f %%" % accurancy_2)
print("3. 정확도 : %f %%" % accurancy_3)
print("4. 정확도 : %f %%" % accurancy_4)
print("5. 정확도 : %f %%" % accurancy_5)
print("평균 : %f %%" % average)


# In[ ]:




