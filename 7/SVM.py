from sklearn.svm import LinearSVC  # 선형 SVM
from sklearn.svm import SVC  # 비선형 SVM
from sklearn.model_selection import train_test_split    # 테스트, 실험 데이터 분리

#from sklearn.model_selection import  cross_val_score





# 파일 읽기
MNIST_image = open('train-images.idx3-ubyte', 'rb')
MNIST_label = open('train-labels.idx1-ubyte', 'rb')

# 1 ~ 4 바이트 : Magic Number 넘김
MNIST_image.read(4)
MNIST_label.read(4)

# Image : 5 ~ 16 바이트 : 이미지 파일 정보
# Big Endian 방식으로 읽어옴
image_amount = int.from_bytes(MNIST_image.read(4), 'big', signed=True)
image_rows = int.from_bytes(MNIST_image.read(4), 'big', signed=True)
image_cols = int.from_bytes(MNIST_image.read(4), 'big', signed=True)

# Label : 5 ~ 8 바이트 : 라벨 파일 정보
label_amount = int.from_bytes(MNIST_label.read(4), 'big', signed=True)

print("[ 파일 정보 ]")
print("Image 개수 : %d" % image_amount)
print("Image 당 행 개수 : %d" % image_rows)
print("Image 당 열 개수 : %d" % image_cols)
print("Label 개수 : %d" % label_amount)

# 데이터 집합
image_data_set = []
label_data_set = []

# 이후, 28x28 바이트(image 1개) 단위로 이미지 파일을,
# 1바이트(label 1개) 단위로 라벨 파일을 읽음
while True:

    # 분류값 1개 읽어옴
    label_read_byte = MNIST_label.read(1)

    # 파일 끝남
    if not label_read_byte:
        break

    label_read = int.from_bytes(label_read_byte, 'big', signed=False)

    # 이미지 1개 읽어옴
    image_read = []
    for i in range(0, 784):
        image_read_pixel = int.from_bytes( MNIST_image.read(1), 'big', signed=False )
        image_read.append(image_read_pixel)


    image_data_set.append(image_read)
    label_data_set.append(label_read)

print()
print("[ 데이터셋 구성 완료 ]")
print("Image Data 개수 : %d" % len(image_data_set))
print("label Data 개수 : %d" % len(label_data_set))





# 학습데이터와 실험데이터 분리
X_train, X_test, y_train, y_test = train_test_split(
    image_data_set, label_data_set, test_size=0.95, random_state=42
)

print()
print("[ 학습데이터, 실험데이터 분리 완료 ]")
print("X_train 개수 : %d" % len(X_train))
print("X_test 개수 : %d" % len(X_test))
print("y_train 개수 : %d" % len(y_train))
print("y_test 개수 : %d" % len(y_test))
print()





# X : image_data_set
# y : label_data_set

# SVM 설정
svm_model_linear = LinearSVC(max_iter=1000000000)
svm_model_nonlinear = SVC(kernel='rbf', C=1, max_iter=1000000000, gamma=0.1)
print("[ svm setting complete ]")

# 학습
train_lin = svm_model_linear.fit(X_train, y_train)
print("[ linear Training complete ]")

train_nlin = svm_model_nonlinear.fit(X_train, y_train)
print("[ nonlinear Training complete ] ")





# 테스트 데이터를 예측
linear_prediction = svm_model_linear.predict(X_test)
print("[ linear Prediction complete ]")

nonlinear_prediction = svm_model_nonlinear.predict(X_test)
print("[ nonlinear Prediction complete ]")


# 정확도 측정
linear_correct = 0
nonlinear_correct = 0
test_amount = len(y_test)

for test_index in range(test_amount):
    if linear_prediction[test_index] == y_test[test_index]:
        linear_correct = linear_correct + 1
    if nonlinear_prediction[test_index] == y_test[test_index]:
        nonlinear_correct = nonlinear_correct + 1

linear_correctness = linear_correct / test_amount
nonlinear_correctness = nonlinear_correct / test_amount

print()
print("Linear SVM 정확도 : %f" % linear_correctness)
print("nonLinear SVM 정확도 : %f" % nonlinear_correctness)




#score = cross_val_score(svm_model_linear, X_train, y_train, cv=3)
#print(len(score))



#print(len(train_lin))
