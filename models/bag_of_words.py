import cv2
import numpy as np
import os
from scipy.cluster.vq import kmeans, vq
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score, f1_score
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb

class BagOfWords:
    def __init__(self, data_folder_path, num_classes, feature_detection, classifier):
        self.num_classes = num_classes
        self.feature_detection = feature_detection
        self.classifier = classifier
        self.RANDOM_STATE = 1337
        self.DATA_FOLDER_PATH = data_folder_path

    def detect_features(self, data):
        feature_detection_algorithm = self.feature_detection

        if feature_detection_algorithm == 'SIFT':
            detector = cv2.SIFT_create()
        elif feature_detection_algorithm == 'ORB':
            detector = cv2.ORB_create()
        else:
            raise Exception("Unknown feature detection algorithm. Accepted values are 'ORB', 'SIFT'.")

        print(f'Detecting features using {feature_detection_algorithm} algorithm.')

        descriptor_list = []
        to_be_removed = []

        for index, row in data.iterrows():
            image_path = os.path.join(self.DATA_FOLDER_PATH, row['Split masked image path'])
            img = cv2.imread(image_path)
            
            if feature_detection_algorithm == 'SIFT':
                img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

            keypoints, descriptor = detector.detectAndCompute(img, None)
            if descriptor is None:
                print(f'Could not detect features for image {image_path}, excluding it from the training data.')
                to_be_removed.append(index)
            else:
                descriptor_list.append((image_path, descriptor))

        descriptors = descriptor_list[0][1]

        if len(to_be_removed > 0):
            print(f"Train data length before excluding images that weren't usable for feature detection: {len(data)}")
            data.drop(to_be_removed, inplace=True)
            print(f"Train data length after excluding images that weren't usable for feature detection: {len(data)}")

        for image_path, descriptor in descriptor_list[1:]:
            descriptors = np.vstack((descriptors, descriptor))

        descriptors_float = descriptors.astype(float)

        k = 200
        voc, variance = kmeans(descriptors_float, k, 1)

        img_features = np.zeros((len(data), k), "float32")
        for i in range(len(data)):
            words, distance = vq(descriptor_list[i][1], voc)
            for w in words:
                img_features[i][w] += 1

        stdslr = StandardScaler().fit(img_features)
        img_features = stdslr.transform(img_features)

        # image features needed for fitting the classifier
        # k, voc and fitted standard scaler needed for prediction
        return {'img_features': img_features, 'k': k, 'voc': voc, 'standard_scaler': stdslr}

    def fit(self, data, img_features):
        classifier = self.classifier

        if classifier == 'RandomForest':
            clf = RandomForestClassifier(random_state=self.RANDOM_STATE)
        elif classifier == 'XGBoost':
            clf = xgb.XGBClassifier(random_state=self.RANDOM_STATE)
        elif classifier == 'SVM':
            clf=LinearSVC(random_state=self.RANDOM_STATE)
        else:
            raise Exception("Unknown classifier. Accepted values are 'SVM', 'RandomForest', 'XGBoost'.")

        clf.fit(img_features, np.array(data['Label']))

        return clf

    def predict(self, data_test, classifier, k, voc, stdslr):
        feature_detection_algorithm = self.feature_detection
        if feature_detection_algorithm == 'SIFT':
            detector = cv2.SIFT_create()
        elif feature_detection_algorithm == 'ORB':
            detector = cv2.ORB_create()
        else:
            raise Exception("Unknown feature detection algorithm. Accepted values are 'ORB', 'SIFT'.")

        descriptor_list_test = []
        to_be_removed_test = []

        for index, row in data_test.iterrows():
            image_path = os.path.join(self.DATA_FOLDER_PATH, row['Split masked image path'])
            img = cv2.imread(image_path)
            if feature_detection_algorithm == 'SIFT':
                img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            keypoints_test, descriptor_test = detector.detectAndCompute(img, None)
            if descriptor_test is None:
                to_be_removed_test.append(index)
            else:
                descriptor_list_test.append((image_path, descriptor_test))

        if len(to_be_removed_test > 0):
            print(f"Test data length before excluding images that weren't usable for feature detection: {len(data_test)}")
            data_test.drop(to_be_removed_test, inplace=True)
            print(f"Test data length after excluding images that weren't usable for feature detection: {len(data_test)}")

        test_features = np.zeros((len(data_test), k), "float32")
        for i in range(len(data_test)):
            words, distance = vq(descriptor_list_test[i][1], voc)
            for w in words:
                test_features[i][w]+= 1

        test_features = stdslr.transform(test_features)

        data_test['pred'] = classifier.predict(test_features)

        accuracy = accuracy_score(data_test['Label'], data_test['pred'])
        f1 = f1_score(data_test['Label'], data_test['pred'], average='weighted')

        return {'predicted_classes': data_test['pred'], 'accuracy': accuracy, 'f1_score': f1}
        
def bag_of_words_classifier(DATA_FOLDER_PATH, num_classes=2, feature_detection='ORB', classifier='SVM') -> BagOfWords:
    _bag_of_words = BagOfWords(DATA_FOLDER_PATH, num_classes, feature_detection, classifier)
    return _bag_of_words