# Evaluation of the OW performance
# We perform Binary Classification in which
# The classifier has to identify the unknown traffic as
# either monitored or unmonitored website

from keras.models import load_model
import numpy as np
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

def Prediction(trained_model = None, dataset = None):
    X_test_Mon = dataset['X_test_Mon'].astype('float32')
    X_test_Unmon = dataset['X_test_Unmon'].astype('float32')
    print "Total testing data ", len(X_test_Mon) + len(X_test_Unmon)
    X_test_Mon = X_test_Mon[:, :, np.newaxis]
    X_test_Unmon = X_test_Unmon[:, :, np.newaxis]
    result_Mon = trained_model.predict(X_test_Mon, verbose=2)
    result_Unmon = trained_model.predict(X_test_Unmon, verbose=2)
    return result_Mon, result_Unmon


def Evaluation(threshold_val = None, monitored_label = None,
                   unmonitored_label = None, result_Mon = None,
                   result_Unmon = None, log_file = None):
    print "Testing with threshold = ", threshold_val
    TP = 0
    FP = 0
    TN = 0
    FN = 0

    # ==============================================================
    # Test with Monitored testing instances
    # evaluation
    for i in range(len(result_Mon)):
        sm_vector = result_Mon[i]
        predicted_class = np.argmax(sm_vector)
        max_prob = max(sm_vector)

        if predicted_class in monitored_label: # predicted as Monitored
            if max_prob >= threshold_val: # predicted as Monitored and actual site is Monitored
                TP = TP + 1
            else: # predicted as Unmonitored and actual site is Monitored
                FN = FN + 1
        elif predicted_class in unmonitored_label: # predicted as Unmonitored and actual site is Monitored
            FN = FN + 1

    # ==============================================================
    # Test with Unmonitored testing instances
    # evaluation
    for i in range(len(result_Unmon)):
        sm_vector = result_Unmon[i]
        predicted_class = np.argmax(sm_vector)
        max_prob = max(sm_vector)

        if predicted_class in monitored_label: # predicted as Monitored
            if max_prob >= threshold_val: # predicted as Monitored and actual site is Unmonitored
                FP = FP + 1
            else: # predicted as Unmonitored and actual site is Unmonitored
                TN = TN + 1
        elif predicted_class in unmonitored_label: # predicted as Unmonitored and actual site is Unmonitored
            TN = TN + 1

    print "TP : ", TP
    print "FP : ", FP
    print "TN : ", TN
    print "FN : ", FN
    print "Total  : ", TP + FP + TN + FN
    TPR = float(TP) / (TP + FN)
    print "TPR : ", TPR
    FPR = float(FP) / (FP + TN)
    print "FPR : ",  FPR
    Precision = float(TP) / (TP + FP)
    print "Precision : ", Precision
    Recall = float(TP) / (TP + FN)
    print "Recall : ", Recall
    print "\n"
    log_file.writelines("%.6f,%d,%d,%d,%d,%.6f,%.6f,%.6f,%.6f\n"%(threshold_val, TP, FP, TN, FN, TPR, FPR, Precision, Recall))

# The evaluation of Open World scenario
def OW_Evaluation():
    evaluation_type = 'OpenWorld_WalkieTalkie'
    print "Evaluation type: ", evaluation_type
    threshold = 1.0 - 1 / np.logspace(0.05, 2, num=15, endpoint=True)
    file_name = '../results/%s.csv'%evaluation_type
    log_file =  open(file_name, "wb")
    # Load data
    dataset = {}
    model_name = ''
    print "Loading data ..."
    from utility import LoadDataWalkieTalkieOW_Evaluation
    X_test_Mon, y_test_Mon, X_test_Unmon, y_test_Unmon = LoadDataWalkieTalkieOW_Evaluation()
    # Load pre-trained model saved from 'Open_World_DF_***_Training.py'
    model_name = '../saved_trained_models/OpenWorld_WalkieTalkie.h5'

    dataset['X_test_Mon'] = X_test_Mon
    dataset['y_test_Mon'] = y_test_Mon
    dataset['X_test_Unmon'] = X_test_Unmon
    dataset['y_test_Unmon'] = y_test_Unmon

    print "Data loaded!"
    print "Loading DF model ..."
    print "The log file will be saved at ", file_name
    print "-- The log file will contains"
    print "-- TP, FP, TN, FN, TPR, FPR, Precision, and Recall for each different threshold"
    print "-- These results will be used to plot the ROC or Precision&Recall Graph"
    trained_model = load_model(model_name)
    print "Model loaded!"
    print "Evaluation Type: ", evaluation_type
    print "Use the model from ", model_name
    result_Mon, result_Unmon = Prediction(trained_model = trained_model, dataset = dataset)
    monitored_label = list(y_test_Mon)
    unmonitored_label = list(y_test_Unmon)
    log_file.writelines("%s,%s,%s,%s,%s,%s  ,%s  ,  %s, %s\n" % ('Threshold', 'TP', 'FP', 'TN', 'FN', 'TPR', 'FPR', 'Precision', 'Recall'))
    for th in threshold:
        Evaluation(threshold_val = th, monitored_label = monitored_label,
                   unmonitored_label = unmonitored_label, result_Mon = result_Mon,
                   result_Unmon = result_Unmon, log_file = log_file)
    log_file.close()

OW_Evaluation()
