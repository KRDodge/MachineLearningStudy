import pandas as pd
import numpy as np
from collections import Counter


class KNN:
    def __init__(self, k):
        self.k = k

    def set(self, data_train, label_train):
        self.data_train = data_train
        self.label_train = label_train

    def predict(self, x_test):
        #테스트 데이터를 모두 유클리드 거리 구하기
        return [self._predict(x) for x in x_test]
    
    def _predict(self, x):
        #모든 데이터의 거리 계산
        distances = [euclidean_distance(x, x_train) for x_train in self.data_train]
        #가장 가까운 k개의 데이터들의 인덱스를 묶어서 저장한다
        k_indices = np.argsort(distances)[:self.k]
        #뽑아낸 index를 바탕으로 가까운 label을 추출
        k_nearest_labels = [self.label_train[i] for i in k_indices]
        #가장 자주 나온 label과 그 개수를 저장
        most_common = Counter(k_nearest_labels).most_common(1)
        #가장 자주나온 label return
        return most_common[0][0]


#유클리드 거리 구하기 
def euclidean_distance(data1, data2):
    return np.sqrt(np.sum(data1 - data2) ** 2)


#파일 읽기 함수
def load_data(file_route):
    data = pd.read_csv(file_route, header=None)
    #데이터 추출
    x = data.iloc[:,:-1].values
    #레이블 추출
    y = data.iloc[:, -1].values
    return x, y


def train_test_split(x, y):
    train_size = 100

    #랜덤 난수 생성, index는 모두 랜덤이 됨
    indices = np.random.permutation(len(x))
    
    #랜덤 중 첫 100개를 학습 데이터로 추출
    train_indices = indices[:train_size]
    #나머지는 테스트 데이터
    test_indices = indices[train_size:]

    #랜덤 index로 train Data, train Label, test Data, test Label 분리
    x_train, x_test = x[train_indices], x[test_indices]
    y_train, y_test = y[train_indices], y[test_indices]

    return x_train, x_test, y_train, y_test



if __name__ == "__main__":
    #코드와 같은 폴더 내의 iris knn 읽기
    file_path = 'iris_KNN.csv'
    #data와 label 분리하여 csv 로드
    x, y = load_data(file_path)

    #학습 데이터, 테스트 데이터, 학습 레이블, 테스트 레이블 분리
    data_train, data_test, label_train, label_test = train_test_split(x,y)

    #k value 조건이 5, 10, 20 ,30
    k_values = [1, 2, 5, 10, 20, 30]
    #결과 출력 위한 리스트
    results = []


    for k_value in k_values:
        #knn 클래스 생성
        knn = KNN(k = k_value)
        #knn 데이터 설정 - knn은 데이터를 그대로 들고 있는게 학습이다.
        knn.set(data_train, label_train)

        #학습 데이터 
        train_predictions = knn.predict(data_train)
        #label의 예측이 맞은 개수 / label의 총 개수를 통해 정확도 판별
        train_accuracy = np.sum(train_predictions == label_train) / len(label_train) * 100

        #테스트 데이터 
        test_predictions = knn.predict(data_test)
        #label의 예측이 맞은 개수 / label의 총 개수를 통해 정확도 판별
        test_accuracy = np.sum(test_predictions == label_test) / len(label_test) * 100
        
        #결과를 저장
        results.append((k_value, train_accuracy, test_accuracy))

    #결과를 k, 학습데이터의 분류율 %, 테스트데이터의 분류율 %로 저장
    results_data_frame = pd.DataFrame(results, columns = ["k", "Train Accuracy (%)", "Test Accuracy (%)"])

    #결과 출력
    print(results_data_frame.to_string(index=False))

