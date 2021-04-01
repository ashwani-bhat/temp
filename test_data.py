import json
import utils
import pandas as pd


def create_entities(dd):
    entities = {}
    for k, v in dd.items():
        for l in v[0]:
            for a, b in l.items():
                if a not in entities.keys():
                    if isinstance(b, list):
                        entities[a] = set(b)
                    else:
                        entities[a] = {b}
                else:
                    if isinstance(b, list):
                        entities[a] |= set(b)
                    else:
                        entities[a] |= {b}
    return entities

def print_dict(key, entities):
    print(f"{utils.OKGREEN} Key:{utils.ENDC} {key} | {utils.WARNING}Value: {utils.ENDC} {entities[key]}")
    print()


def original_annotation_check():
    path = 'data/original_annotation/'
    f = open(path + 'dailydialog_train.json')
    dd = json.load(f)

    entities = create_entities(dd)
    key = ["turn", "type", "speaker", "emotion"]
    for k in key:
        print_dict(k, entities)

def classification_dataset_check():
    data_path = '/content/drive/MyDrive/NLP projects/Emotion-cause-pair/RECCON/data/classification/fold1/dailydialog_classification_test_with_context.csv'

    df = pd.read_csv(data_path)
    print(df.iloc[1000].values)

def check_len():
    data_path_test = 'data/qa/fold3/dailydialog_qa_test_with_context.json'
    data_path_valid = 'data/qa/fold3/dailydialog_qa_valid_with_context.json'
    data_path_train = 'data/qa/fold3/dailydialog_qa_train_with_context.json'
    data_path_iemocap = 'data/qa/fold3/iemocap_qa_test_with_context.json'

    f = open(data_path_test)
    print("Test data length: ", len(json.load(f)))

    f = open(data_path_valid)
    print("Valid data length: ", len(json.load(f)))
    
    f = open(data_path_train)
    print("Train data length: ", len(json.load(f)))
    
    f = open(data_path_iemocap)
    print("Iemocap test data length: ", len(json.load(f)))
    
if __name__ == '__main__':
    check_len()