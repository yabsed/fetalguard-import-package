
from utils import *
import argparse
import warnings
warnings.filterwarnings("ignore")
random_state = 42

# input ID from user
parser = argparse.ArgumentParser()
parser.add_argument('--ID', type = str, default = '10001_0', 
                    help='대상자 1명의 ID를 입력하세요. 태아 상태 진단 모델이 응급, 비응급 상태를 예측합니다. \n ex) 10001_0')
args = parser.parse_args()
input_ID = args.ID  


current_path = os.path.dirname(os.path.realpath(__file__)) # current_path
yaml_path = os.path.join(current_path, 'data_path.yaml') # dataset path
model_dir = os.path.join(current_path, 'model', 'xgb_clf.model') # saved model pathn


def return_true_pred_label(yaml_path, id):
    # make 1-row dataframe
    label_dir, emr_dir, anno_dir = get_id_dir(yaml_path, id)
    one_df = get_one_df(id, label_dir, emr_dir, anno_dir)
    one_df = one_df.astype({ 'Emergency' : 'int' })
    
    #get true label 
    true_label = one_df.at[0, 'Emergency'] 
    # preprocesing
    X_test = preprocess_ID_df(one_df) 
    # model load
    new_xgb = XGBClassifier() # model initialization
    new_xgb.load_model(model_dir) # load model
    # prediction
    y_pred = new_xgb.predict(X_test)
    pred_label = y_pred[0]  
    
    #print the result of prediction
    if pred_label ==  0: # negative
        pred_state = '응급 상황이 아닙니다.'
    elif pred_label == 1: # positive
        pred_state = '응급 상황입니다.'
    else:
        print('유효하지 않은 상태입니다. 예측 라벨을 확인하세요.')
    
    print('\n============태아 상태 진단 결과=============\n')
    print('예측 대상자 ID : ', id)
    print('태아 상태 예측 결과 :', pred_state)
    
    return true_label, pred_label
         
if __name__ == "__main__":
    _, pred_label = return_true_pred_label(yaml_path, id = input_ID)




