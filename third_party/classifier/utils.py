
import pandas as pd
import yaml
import numpy as np
from xgboost import XGBClassifier
from sklearn.metrics import *
from sklearn.model_selection import *
import json
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.io as pio
import re
import os
from datetime import datetime
from statistics import mean, median
import warnings
warnings.filterwarnings("ignore")
random_state = 42

# ------------------------------------------------------ make csv -------------------------------------------------------- 
def get_fhr_toco(anno, var): # var 자리에 fhr, toco를 각각 넣으면 된다. 
  data = anno['data']
  variable = ''
  for i in range(len(data)):
    i_var = data[i][var]
    variable += i_var

  numbers = re.findall(r'\d+', variable) # fhr or toco의 전체 item들이 str 타입으로 리스트에 들어가있음. 
  int_var = list(map(int, numbers)) # 각 요소들을 전부 int type으로 형변환 시켜줌. 
  return int_var

def get_prop_abnormal(str): # get proportion of 'abnormal' in feature 'Abnormality'
    list = str.split('_')
    prop = list.count('1')/len(list)
    return round(prop,  1)

def return_statics(lst): # list를 받아서 min, max, mean, median 반환해주는 함수
    return min(lst), max(lst),  int(median(lst)) , int(mean(lst))

def make_df_from_jsons(label_dir, emr_dir, anno_dir, json_ID):
    #1. label/emr/anno i 번째 json 열기
    label_path = label_dir + json_ID
    emr_path = emr_dir + json_ID
    anno_path = anno_dir + json_ID

    # read each json file
    with open(label_path, "r") as f:
        label = json.load(f)

    with open(emr_path, "r") as f:
        emr = json.load(f)

    with open(anno_path, "r") as f:
        anno = json.load(f)


    # dict to dataframe
    label = pd.DataFrame([label])
    emr = pd.DataFrame([emr])

    # annotation json에서 fhr, toco만 가져오기. 
    fhr = get_fhr_toco(anno, 'fhr')
    toco = get_fhr_toco(anno, 'toco')

    # label, emr 부터 ID를 기준으로 inner join 하기 
    merge_df = pd.merge(emr, label)
    
    # merge된 dataframe에 마지막으로 fhr, toco  column 생성해주기. 
    merge_df['FHR'] =[fhr]
    merge_df['TOCO'] = [toco]
    
    # FHR, TOCO list로 부터 통계값 파생 변수 생성
    merge_df['min_fhr'], merge_df['max_fhr'], merge_df['median_fhr'], merge_df['mean_fhr'] = zip(*merge_df['FHR'].apply(lambda x : return_statics(x)))
    merge_df['min_toco'], merge_df['max_toco'], merge_df['median_toco'], merge_df['mean_toco'] = zip(*merge_df['TOCO'].apply(lambda x : return_statics(x)))
    
    return merge_df

# 개별 json 파일을 하나의 csv 파일로 합치는 함수
def get_total_df(label_dir, emr_dir, anno_dir):
    label_list = os.listdir(label_dir) 
    total_len = len(label_list) # 총 json 파일 개수
    
    tmp_df = pd.DataFrame() #빈 데이터프레임(tmp_df) 만들고 거기에 계속해서 덧붙임.

    for i in range(total_len):
        
        ith_json_ID = label_list[i] # i번째 json 파일 이름 가져오기.
        ith_df = make_df_from_jsons(label_dir, emr_dir, anno_dir, json_ID = ith_json_ID)
        tmp_df = pd.concat([tmp_df, ith_df])
    
    return tmp_df
        
# 한 사람에 대한 예측 할 때 사용.
def get_one_df(ID, label_dir, emr_dir, anno_dir):
    json_ID = ID + '.json' # ex) json_ID = 10001_0.json
    one_df = make_df_from_jsons(label_dir, emr_dir, anno_dir, json_ID)
    return one_df 
       
# ----------------------------------------------------- preprocess -------------------------------------------------------
def first_step_of_preprocess(df):
    
    original_df_len = len(df) # length of data before preprocessing
    
    # convert str or object to int
    col_list = ['Mother.GHTN', 'Mother.Hypertension', 'Mother.GDM', 'Mother.DM', 'Mother.pre-eclampsia', 'Delivery', 
                  'FetalDistress', 'FGR', 'Placenta.Complication', 'Sex', 'Jaundice', 'prematurity', 'LBW', 'Anomaly', 
                  'Anomaly1', 'Anomaly2', 'Anomaly3', 'Anomaly4', 'Anomaly5', 'Anomaly6', 'Anomaly7', 'Anomaly8', 'twins']
    
    for col in col_list:
        df[col] =  pd.to_numeric(df[col])
        
    df = df.drop(df.iloc[:, 1:4], axis = 1) #drop Mother.de-identification_ID~ MEASURE_DATE
    df = df.drop(df.iloc[:, 29:35], axis = 1) #drop APGAR.1min ~ UA.pCO2
    df = df.drop(df.iloc[:, 45:54], axis = 1) #drop BaseLine~CA
    
    drop_columns = ['Father.Birth Date','Mother.ABO type', 'Mother.RH type',
                    'Placenta.Weight', 'Cervix','Intubation' ,'Fetal_Monitor','Bbox',
                    'FHR', 'TOCO', 'NICU.Adm']
    
    df = df.drop(drop_columns, axis = 1) #unknown이 많은 열 삭제 (by column name)

    #2. age column 생성 from Birth Date
    df['Mother.age'] =  (df['Birth Date'] // 100) - (df['Mother.Birth Date'] // 100) + 1
    
    #3.remove birth date 
    df = df.drop(['Mother.Birth Date', 'Birth Date'], axis = 1) 
    
    return original_df_len, df
    
def preprocess(df, dtype):
    
    _, df = first_step_of_preprocess(df)
   
    #unknown 있는 행 제거
    # 9999   or '9999'를 nan으로 맵핑하기. --> 그 후, dropna로 제거
    df = df.replace(9999, np.NaN)
    df = df.replace('9999', np.NaN)
    df = df.dropna()
        

    # prop_abnormal column  생성 
    if dtype == 'train' or 'val' or 'test':
        df['prop_abnormal'] = df['Abnormality'].apply(get_prop_abnormal)
        
    else: print('Invalid dtype. Enter "train" or "val" or "test".')
        
    # test 
    df = df.drop(['Abnormality'], axis = 1)  # 사용한 변수 제거 
    
    return df


def preprocess_ID_df(df):
    
    _, df = first_step_of_preprocess(df)
    # unknown 있는지 체크
    df = df.replace(9999, np.NaN)
    df = df.replace('9999', np.NaN)
    total_na_num = df.isnull().sum().sum()
    if total_na_num != 0:
        print('unknown이 발생했습니다. 전처리 함수를 종료합니다.')
        exit() # 전처리함수 종료
    
    # 이상 시점 비율 변수 생성
    df['prop_abnormal'] = df['Abnormality'].apply(get_prop_abnormal)
    
    # X 추출
    X = df.drop(['ID','Emergency','Abnormality'], axis = 1) # 타겟변수 'Emergency'제거 
    
    return X
    
 
def save_preprocessed_csv(yaml_path, save_dir, dtype):
    with open(yaml_path) as f:
        paths = yaml.load(f, Loader = yaml.FullLoader)
        
    if dtype == 'train':
        path = paths['train']
    elif dtype == 'val':
        path = paths['val']
    elif dtype == 'test':
        path = paths['test']
    else:
        print('Invalid dtype. Expected "train" or "val" or "test".')
    
    label_dir = path['label']
    emr_dir = path['emr']
    anno_dir = path['annotation']
    
    df = get_total_df(label_dir, emr_dir, anno_dir)
    df = df.astype({ 'Emergency' : 'int' }) #  'str' to 'int'

    after_preprocess_df = preprocess(df, dtype=dtype)
    
    # 전처리 완료한 데이터프레임을 csv로 저장. 
    csv_save_dir = f'{save_dir}/{dtype}.csv'
    after_preprocess_df.to_csv(csv_save_dir, mode='w', index = False)
# ------------------------------------------------------ train and eval --------------------------------------------------- 

def return_X_and_y(df):
    X = df.drop(['Emergency'], axis = 1)
    y = df[['Emergency']]
    return X, y

# 필수 파라미터 : X_train, y_train, save_dir
def xgb_train_save(X_train, y_train, save_dir, params=None, 
                    X_test = None, y_test = None, 
                    set_params = False,
                    plotting = False, random_state=42):
    
    # ID column 제거
    X_train = X_train.drop(['ID'], axis = 1)
    if set_params == False: # use default parameters
        model = XGBClassifier(random_state=random_state)
    
    # 사용자가 모델의 파라미터를 변경할 경우 
    else: # set parametesrs from input 'params' dictionary
        model = XGBClassifier(n_estimators=params['n_estimators'], 
                          learning_rate=params['learning_rate'], 
                          max_depth=params['max_depth'], 
                          gamma=params['gamma'], 
                          scale_pos_weight=params['scale_pos_weight'],
                          random_state=random_state)
    
    
    print('MODEL TRAINING START . > . > . > . > . > . > . > . >')
    model.fit(X_train, y_train)

    # model save
    model_path = f'{save_dir}xgb_clf.model'
    model.save_model(model_path)
    print('MODEL SUCCESSFULLY SAVED.')
    # plot 변수 중요도 저장
    if plotting == True:
        plot_feature_importance(model, X_train.columns) # 실행되는 폴더 내에 이미지가 생성됨. 
    
    #  테스트셋 입력할 경우 평가 매트릭 출력
    if (X_test == True) and (y_test == True):
        xgb_eval(X_test, y_test, model)
        
    
def xgb_eval(X_test, y_test, model_path, log_dir ='./', plot_roc_curve = False):
    
    start = datetime.now()
    # 모델 초기화 및 입력 받은 경로의 모델 불러오기.
    model = XGBClassifier()
    model.load_model(model_path)

    # prediction
    # ID 별도 저장후 제거
    ID = list(X_test['ID'])
    X_test = X_test.drop(['ID'], axis = 1)
    # prediction
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:,1]
    y_pred_proba = np.round(y_pred_proba, 3) #소수점 반올림

    # prediction 로그 데이터프레임 생성
    y_test = list(y_test['Emergency'])
    pred_df = pd.DataFrame(list(zip(ID, y_test, y_pred, y_pred_proba)), 
                           columns = ['    ID   ', 'True', 'Pred', 'Pred_proba'])
    pred_df['Pred_proba'] = pred_df['Pred_proba'].round(3) # proba 소수점 반올림
    pred_df.to_csv('./tmp.txt', sep = '\t') # 임시 저장
    
    # use confusion matrix library : "classificatin_report"
    print('\n================== Classification Report ===================\n')
    clf_report = classification_report(y_test, y_pred, labels=[0, 1], digits = 3)
    print(clf_report)
    print('\n================== Confusion Matrix ========================')
    print(confusion_matrix(y_test, y_pred))    
    
    # end time
    end =  datetime.now()
    # 2차(최종) evaluation 끝난 후 log 파일 생성
    log_path = os.path.join(log_dir, 'evaluation_log.txt')
    with open(log_path, 'w') as f:
        # evaluation 실행 명령어
        eval_call = 'python eval.py'
        f.write(f'Start\t{start}\t{eval_call}\n\n') # 실행 시작 시각
        # 예측 기록
        with open('./tmp.txt') as df:
            for line in df:
                f.write(line)
        f.write('\n')
        f.write(clf_report) # classification report
        f.write(f'\nEnd\t{end}\n') # 실행 끝난 시각
        f.close()
    
    # 임시 df.txt 파일 제거
    tmp = './tmp.txt'
    if os.path.isfile(tmp):
        os.remove(tmp)
        
    #plotly roc curve save
    if plot_roc_curve == True:
        plot_roc_curve_plotly(y_test, y_pred, y_pred_proba)
        plot_roc_curve_plt(y_test, y_pred, y_pred_proba)
    
    
def grid_search(param_grid, X_train, y_train, X_test, y_test):
    print('Starting GridSearch . . .')
    grid_search = GridSearchCV(XGBClassifier(),  
                                param_grid, 
                                cv = 5,
                                scoring = 'f1_macro',
                                return_train_score= True,
                                n_jobs = -1,
                                
                                )
    # run GridSearch
    grid_search.fit(X_train, y_train)
    print('End of grid_search fitting')

    # prediction
    estimator = grid_search.best_estimator_
    
    y_pred = estimator.predict(X_test)
    print(classification_report(y_test, y_pred, labels=[0, 1], digits = 3))
    print('\n===== Classification Report ====')
    print(confusion_matrix(y_test, y_pred))  

    # best parmameters
    print('Best Params')
    print(grid_search.best_params_)
    print('Best Score')
    print(grid_search.best_score_)

#_____________________________________________ prediction _____________________________________________________
# 
# id, yaml_path을 입력받아 해당 id의 json파일 경로를 찾는 함수.
def get_id_dir(yaml_path, id):
    
    #yaml file open
    with open(yaml_path) as f:
        paths = yaml.load(f, Loader = yaml.FullLoader)
        
    train_path = paths['train']
    val_path = paths['val']
    test_path = paths['test']
    
    train_label_lst = os.listdir(train_path['label'])
    val_label_lst = os.listdir(val_path['label'])
    test_label_lst = os.listdir(test_path['label'])
    
    json_id = id + '.json'
    
    if json_id in train_label_lst:
        where_id = 'train'
    elif json_id in val_label_lst:
        where_id = 'val'
    elif json_id in test_label_lst:
        where_id = 'test'
    else: 
        print('유효하지 않은 ID입니다. train, val, test set 어느 한 곳에 있는 ID를 입력하세요.')
        
    
    label_dir = paths[where_id]['label']
    emr_dir = paths[where_id]['emr']
    anno_dir = paths[where_id]['annotation']
    
    return label_dir, emr_dir, anno_dir


# ------------------------------------------------------ plot utils ------------------------------------------------------- 

# 변수 중요도 저장 함수
def plot_feature_importance(model, cols):
    FIM_df = pd.DataFrame(zip(cols, model.feature_importances_), 
                          columns = ['Feature', 'FIM'])
    FIM_df = FIM_df.sort_values('FIM', ascending = False).head(20) # top 20 columns by FIM.
    fig = px.bar(FIM_df, y='FIM', x='Feature', text_auto='.1s',
                title=" Feature Importances")
    fig.update_traces(textfont_size=9, textangle=0, textposition="outside", cliponaxis=False)
    pio.write_image(fig, file = './FIM_fig.png', format='png')
    
# ROC curve graph 저장 함수
def plot_roc_curve_plotly(y_test, y_pred,  y_pred_proba):
    # plot AUC graph
    fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
    fig = px.area(
        x=fpr, y=tpr,
        title=f'ROC Curve (AUC={roc_auc_score(y_test, y_pred):.4f})',
        labels=dict(x='False Positive Rate', y='True Positive Rate'),
        width=700, height=500
    )
    fig.add_shape(
        type='line', line=dict(dash='dash'),
        x0=0, x1=1, y0=0, y1=1
    )

    fig.update_yaxes(scaleanchor="x", scaleratio=1)
    fig.update_xaxes(constrain='domain')
    
    pio.write_image(fig, file = './plotly_roc_curve.png', format='png')
        
    
def plot_roc_curve_plt(y_test, y_pred, y_pred_proba):
    # plot AUC graph
    fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
    auc = np.round(roc_auc_score(y_test, y_pred), 3)
    plt.plot(fpr, tpr, label=f'ROC curve (area = {auc})', color ='#ff7f0e')
    plt.plot([0,1], [0, 1], 'k--') # 가운데 대각선 라인
    
    #start , end = plt.xlim()
    #plt.xticks(np.round(np.arange(start, end, 0.2),2))
    plt.xlim(0,1)
    plt.ylim(0,1)
    plt.grid()
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.legend(loc='lower right')
    plt.title('XGBoost ROC curve')
    plt.show()
    plt.savefig('./xgb_roc_curve.png')










