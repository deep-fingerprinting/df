import pickle
import numpy as np
import os

def LoadDataSetFromRawTraces(dir, pktSeqLen):
    files = os.listdir(dir)
    files = [f for f in files if os.path.isfile(dir+'/'+f)]

    data = []
    labels = []
    for f in files:
        fn = f.split(".")[0]
        if "-" in fn:
            label = (int(fn.split("-")[0]))
        else:
            continue
        
        seq = []
        with open(dir+'/'+f, "r") as fd:
            for line in fd.readlines():
                content = line.split()
                if len(content) > 0:
                    if content[1].startswith("-"):
                        seq.append(-1)
                    else:
                        seq.append(1)
        
        deficit = pktSeqLen - len(seq)
        if deficit > 0:
            for i in range(0, deficit):
                seq.append(0)
        elif deficit < 0:
            seq = seq[0:pktSeqLen]

        data.append(seq)
        labels.append(label)

    return data, labels

def LoadTsDataSetFromRawTraces(dir, pktSeqLen):
    files = os.listdir(dir)
    files = [f for f in files if os.path.isfile(dir+'/'+f)]

    data = []
    labels = []
    for f in files:
        fn = f.split(".")[0]
        if "-" in fn:
            label = (int(fn.split("-")[0]))
        else:
            continue
        
        seq = []
        last_ts = 0
        with open(dir+'/'+f, "r") as fd:
            count = 0
            for line in fd.readlines():
                content = line.split()
                ts = float(content[0])
                if count == 0:
                    last_ts = ts
                    ts = 0
                else:
                    ts = ts - last_ts
                    last_ts = float(content[0])
                if len(content) > 0:
                    if content[1].startswith("-"):
                        seq.append(-ts)
                    else:
                        seq.append(ts)

                count += 1
        
        deficit = pktSeqLen - len(seq)
        if deficit > 0:
            for i in range(0, deficit):
                seq.append(0)
        elif deficit < 0:
            seq = seq[0:pktSeqLen]

        data.append(seq)
        labels.append(label)

    return data, labels

# Load data for non-defended dataset for CW setting
def LoadDataNoDefCW():

    print("Loading non-defended dataset for closed-world scenario")
    # Point to the directory storing data
    dataset_dir = '../dataset/ClosedWorld/NoDef/'

    # X represents a sequence of traffic directions
    # y represents a sequence of corresponding label (website's label)

    # Load training data
    with open(dataset_dir + 'X_train_NoDef.pkl', 'rb') as handle:
        X_train = np.array(pickle.load(handle, encoding="UTF-8"))
    with open(dataset_dir + 'y_train_NoDef.pkl', 'rb') as handle:
        y_train = np.array(pickle.load(handle))

    # Load validation data
    with open(dataset_dir + 'X_valid_NoDef.pkl', 'rb') as handle:
        X_valid = np.array(pickle.load(handle))
    with open(dataset_dir + 'y_valid_NoDef.pkl', 'rb') as handle:
        y_valid = np.array(pickle.load(handle))

    # Load testing data
    with open(dataset_dir + 'X_test_NoDef.pkl', 'rb') as handle:
        X_test = np.array(pickle.load(handle))
    with open(dataset_dir + 'y_test_NoDef.pkl', 'rb') as handle:
        y_test = np.array(pickle.load(handle))

    print("Data dimensions:")
    print("X: Training data's shape : ", X_train.shape)
    print("y: Training data's shape : ", y_train.shape)
    print("X: Validation data's shape : ", X_valid.shape)
    print("y: Validation data's shape : ", y_valid.shape)
    print("X: Testing data's shape : ", X_test.shape)
    print("y: Testing data's shape : ", y_test.shape)

    return X_train, y_train, X_valid, y_valid, X_test, y_test

# Load data for non-defended dataset for CW setting
def LoadDataWTFPADCW():

    print("Loading WTF-PAD dataset for closed-world scenario")
    # Point to the directory storing data
    dataset_dir = '../dataset/ClosedWorld/WTFPAD/'

    # X represents a sequence of traffic directions
    # y represents a sequence of corresponding label (website's label)

    # Load training data
    with open(dataset_dir + 'X_train_WTFPAD.pkl', 'rb') as handle:
        X_train = np.array(pickle.load(handle))
    with open(dataset_dir + 'y_train_WTFPAD.pkl', 'rb') as handle:
        y_train = np.array(pickle.load(handle))

    # Load validation data
    with open(dataset_dir + 'X_valid_WTFPAD.pkl', 'rb') as handle:
        X_valid = np.array(pickle.load(handle))
    with open(dataset_dir + 'y_valid_WTFPAD.pkl', 'rb') as handle:
        y_valid = np.array(pickle.load(handle))

    # Load testing data
    with open(dataset_dir + 'X_test_WTFPAD.pkl', 'rb') as handle:
        X_test = np.array(pickle.load(handle))
    with open(dataset_dir + 'y_test_WTFPAD.pkl', 'rb') as handle:
        y_test = np.array(pickle.load(handle))

    print("Data dimensions:")
    print("X: Training data's shape : ", X_train.shape)
    print("y: Training data's shape : ", y_train.shape)
    print("X: Validation data's shape : ", X_valid.shape)
    print("y: Validation data's shape : ", y_valid.shape)
    print("X: Testing data's shape : ", X_test.shape)
    print("y: Testing data's shape : ", y_test.shape)

    return X_train, y_train, X_valid, y_valid, X_test, y_test

# Load data for non-defended dataset for CW setting
def LoadDataWalkieTalkieCW():

    print("Loading Walkie-Talkie dataset for closed-world scenario")
    # Point to the directory storing data
    dataset_dir = '../dataset/ClosedWorld/WalkieTalkie/'

    # X represents a sequence of traffic directions
    # y represents a sequence of corresponding label (website's label)

    # Load training data
    with open(dataset_dir + 'X_train_WalkieTalkie.pkl', 'rb') as handle:
        X_train = np.array(pickle.load(handle))
    with open(dataset_dir + 'y_train_WalkieTalkie.pkl', 'rb') as handle:
        y_train = np.array(pickle.load(handle))

    # Load validation data
    with open(dataset_dir + 'X_valid_WalkieTalkie.pkl', 'rb') as handle:
        X_valid = np.array(pickle.load(handle))
    with open(dataset_dir + 'y_valid_WalkieTalkie.pkl', 'rb') as handle:
        y_valid = np.array(pickle.load(handle))

    # Load testing data
    with open(dataset_dir + 'X_test_WalkieTalkie.pkl', 'rb') as handle:
        X_test = np.array(pickle.load(handle))
    with open(dataset_dir + 'y_test_WalkieTalkie.pkl', 'rb') as handle:
        y_test = np.array(pickle.load(handle))

    print("Data dimensions:")
    print("X: Training data's shape : ", X_train.shape)
    print("y: Training data's shape : ", y_train.shape)
    print("X: Validation data's shape : ", X_valid.shape)
    print("y: Validation data's shape : ", y_valid.shape)
    print("X: Testing data's shape : ", X_test.shape)
    print("y: Testing data's shape : ", y_test.shape)

    return X_train, y_train, X_valid, y_valid, X_test, y_test

# Load data for non-defended dataset for OW training
def LoadDataNoDefOW_Training():

    print("Loading non-defended dataset for open-world scenario for training")
    # Point to the directory storing data
    dataset_dir = '../dataset/OpenWorld/NoDef/'

    # X represents a sequence of traffic directions
    # y represents a sequence of corresponding label (website's label)

    # Load training data
    with open(dataset_dir + 'X_train_NoDef.pkl', 'rb') as handle:
        X_train = np.array(pickle.load(handle))
    with open(dataset_dir + 'y_train_NoDef.pkl', 'rb') as handle:
        y_train = np.array(pickle.load(handle))

    # Load validation data
    with open(dataset_dir + 'X_valid_NoDef.pkl', 'rb') as handle:
        X_valid = np.array(pickle.load(handle))
    with open(dataset_dir + 'y_valid_NoDef.pkl', 'rb') as handle:
        y_valid = np.array(pickle.load(handle))


    print("Data dimensions:")
    print("X: Training data's shape : ", X_train.shape)
    print("y: Training data's shape : ", y_train.shape)
    print("X: Validation data's shape : ", X_valid.shape)
    print("y: Validation data's shape : ", y_valid.shape)

    return X_train, y_train, X_valid, y_valid

# Load data for non-defended dataset for OW evaluation
def LoadDataNoDefOW_Evaluation():

    print("Loading non-defended dataset for open-world scenario for evaluation")
    # Point to the directory storing data
    dataset_dir = '../dataset/OpenWorld/NoDef/'

    # X represents a sequence of traffic directions
    # y represents a sequence of corresponding label (website's label)

    # Load training data
    with open(dataset_dir + 'X_test_Mon_NoDef.pkl', 'rb') as handle:
        X_test_Mon = pickle.load(handle)
    with open(dataset_dir + 'y_test_Mon_NoDef.pkl', 'rb') as handle:
        y_test_Mon = pickle.load(handle)
    with open(dataset_dir + 'X_test_Unmon_NoDef.pkl', 'rb') as handle:
        X_test_Unmon = pickle.load(handle)
    with open(dataset_dir + 'y_test_Unmon_NoDef.pkl', 'rb') as handle:
        y_test_Unmon = pickle.load(handle)

    X_test_Mon = np.array(X_test_Mon)
    y_test_Mon = np.array(y_test_Mon)
    X_test_Unmon = np.array(X_test_Unmon)
    y_test_Unmon = np.array(y_test_Unmon)

    return X_test_Mon, y_test_Mon, X_test_Unmon, y_test_Unmon

# Load data for WTF-PAD dataset for OW training
def LoadDataWTFPADOW_Training():

    print("Loading WTF-PAD dataset for open-world scenario for training")
    # Point to the directory storing data
    dataset_dir = '../dataset/OpenWorld/WTFPAD/'

    # X represents a sequence of traffic directions
    # y represents a sequence of corresponding label (website's label)

    # Load training data
    with open(dataset_dir + 'X_train_WTFPAD.pkl', 'rb') as handle:
        X_train = np.array(pickle.load(handle))
    with open(dataset_dir + 'y_train_WTFPAD.pkl', 'rb') as handle:
        y_train = np.array(pickle.load(handle))

    # Load validation data
    with open(dataset_dir + 'X_valid_WTFPAD.pkl', 'rb') as handle:
        X_valid = np.array(pickle.load(handle))
    with open(dataset_dir + 'y_valid_WTFPAD.pkl', 'rb') as handle:
        y_valid = np.array(pickle.load(handle))


    print("Data dimensions:")
    print("X: Training data's shape : ", X_train.shape)
    print("y: Training data's shape : ", y_train.shape)
    print("X: Validation data's shape : ", X_valid.shape)
    print("y: Validation data's shape : ", y_valid.shape)

    return X_train, y_train, X_valid, y_valid

# Load data for WTF-PAD dataset for OW evaluation
def LoadDataWTFPADOW_Evaluation():

    print("Loading WTF-PAD dataset for open-world scenario for evaluation")
    # Point to the directory storing data
    dataset_dir = '../dataset/OpenWorld/WTFPAD/'

    # X represents a sequence of traffic directions
    # y represents a sequence of corresponding label (website's label)

    # Load training data
    with open(dataset_dir + 'X_test_Mon_WTFPAD.pkl', 'rb') as handle:
        X_test_Mon = pickle.load(handle)
    with open(dataset_dir + 'y_test_Mon_WTFPAD.pkl', 'rb') as handle:
        y_test_Mon = pickle.load(handle)
    with open(dataset_dir + 'X_test_Unmon_WTFPAD.pkl', 'rb') as handle:
        X_test_Unmon = pickle.load(handle)
    with open(dataset_dir + 'y_test_Unmon_WTFPAD.pkl', 'rb') as handle:
        y_test_Unmon = pickle.load(handle)

    X_test_Mon = np.array(X_test_Mon)
    y_test_Mon = np.array(y_test_Mon)
    X_test_Unmon = np.array(X_test_Unmon)
    y_test_Unmon = np.array(y_test_Unmon)

    return X_test_Mon, y_test_Mon, X_test_Unmon, y_test_Unmon

# Load data for WalkieTalkie dataset for OW training
def LoadDataWalkieTalkieOW_Training():

    print("Loading Walkie-Talkie dataset for open-world scenario for training")
    # Point to the directory storing data
    dataset_dir = '../dataset/OpenWorld/WalkieTalkie/'

    # X represents a sequence of traffic directions
    # y represents a sequence of corresponding label (website's label)

    # Load training data
    with open(dataset_dir + 'X_train_WalkieTalkie.pkl', 'rb') as handle:
        X_train = np.array(pickle.load(handle))
    with open(dataset_dir + 'y_train_WalkieTalkie.pkl', 'rb') as handle:
        y_train = np.array(pickle.load(handle))

    # Load validation data
    with open(dataset_dir + 'X_valid_WalkieTalkie.pkl', 'rb') as handle:
        X_valid = np.array(pickle.load(handle))
    with open(dataset_dir + 'y_valid_WalkieTalkie.pkl', 'rb') as handle:
        y_valid = np.array(pickle.load(handle))


    print("Data dimensions:")
    print("X: Training data's shape : ", X_train.shape)
    print("y: Training data's shape : ", y_train.shape)
    print("X: Validation data's shape : ", X_valid.shape)
    print("y: Validation data's shape : ", y_valid.shape)

    return X_train, y_train, X_valid, y_valid

# Load data for WTF-PAD dataset for OW evaluation
def LoadDataWalkieTalkieOW_Evaluation():

    print("Loading Walkie-Talkie dataset for open-world scenario for evaluation")
    # Point to the directory storing data
    dataset_dir = '../dataset/OpenWorld/WalkieTalkie/'

    # X represents a sequence of traffic directions
    # y represents a sequence of corresponding label (website's label)

    # Load training data
    with open(dataset_dir + 'X_test_Mon_WalkieTalkie.pkl', 'rb') as handle:
        X_test_Mon = pickle.load(handle)
    with open(dataset_dir + 'y_test_Mon_WalkieTalkie.pkl', 'rb') as handle:
        y_test_Mon = pickle.load(handle)
    with open(dataset_dir + 'X_test_Unmon_WalkieTalkie.pkl', 'rb') as handle:
        X_test_Unmon = pickle.load(handle)
    with open(dataset_dir + 'y_test_Unmon_WalkieTalkie.pkl', 'rb') as handle:
        y_test_Unmon = pickle.load(handle)

    X_test_Mon = np.array(X_test_Mon)
    y_test_Mon = np.array(y_test_Mon)
    X_test_Unmon = np.array(X_test_Unmon)
    y_test_Unmon = np.array(y_test_Unmon)

    return X_test_Mon, y_test_Mon, X_test_Unmon, y_test_Unmon