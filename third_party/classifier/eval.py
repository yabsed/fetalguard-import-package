#%%
import pandas as pd
from utils import *
import warnings
warnings.filterwarnings("ignore")
random_state = 42

#load test csv
current_path = os.path.dirname(os.path.realpath(__file__)) #current directory path
test_path = os.path.join(current_path, 'data/test.csv') 

# set dir for evaluation log record
log_dir = os.path.join(current_path, 'log') 
os.makedirs(log_dir, exist_ok = True) #실행 경로 하위에 'log' 폴더 생성

#load saved model
model_path = os.path.join(current_path, 'model', 'xgb_clf.model') # saved model path

test = pd.read_csv(test_path) #load csv as dataframe
X_test, y_test = return_X_and_y(test) # get X and y

# run evaluation function from utils.py
xgb_eval(X_test, y_test, 
        model_path = model_path, 
        log_dir = log_dir,
        plot_roc_curve = False) #  If you want to plot roc curve, set this to True


# %%
