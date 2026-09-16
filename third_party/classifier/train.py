
import pandas as pd
from utils import *
import warnings
warnings.filterwarnings("ignore")
random_state = 42

current_path = os.path.dirname(os.path.realpath(__file__)) # upper directory of "train.csv"
model_save_dir = os.path.join(current_path, 'model/') # set model saving path
train_path = os.path.join(current_path, 'data/train.csv')  # load train csv

train = pd.read_csv(train_path) # read csv as dataframe
X_train, y_train = return_X_and_y(train)

# set parameters
params={'n_estimators': 200,
        'learning_rate': .3,
        'max_depth': 3,
        'gamma': .5,
        'scale_pos_weight': 1 }

set_params = False #  ** if you want to change the parameters, set this to "True" ** 

# run xgb model train and save function
xgb_train_save(X_train, y_train, 
            save_dir = model_save_dir, 
            params = params,
            set_params = set_params,
            random_state = random_state)

