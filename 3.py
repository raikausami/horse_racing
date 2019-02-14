import pandas as pd
import numpy as np
import datetime as dt
import math
from sklearn.tree import DecisionTreeClassifier, export_graphviz
from graphviz import Source
from sklearn.externals.six import StringIO
from sklearn import tree
import pydotplus
from sklearn.tree import DecisionTreeRegressor
from graphviz import Digraph
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import RandomForestClassifier
from sklearn import cross_validation
import pydot


filename = "g1.csv"

df = pd.read_csv(filename,header=0,encoding="Shift_JISx0213")
X_cols=["umaban","horse_weight","zougen","age","weight","jockey","male","female","gelding","distance","frame_num","condition","past1_result","past2_result","past3_result","total_game","win1","win2","win3"]
y_cols=["result"]
X = df[X_cols].as_matrix().astype('float')
y = df[y_cols].as_matrix().astype('int').flatten()


X_train, X_test, y_train, y_test = cross_validation.train_test_split(X, y, test_size=0.3)
forest = RandomForestClassifier(n_estimators = 100)

forest = forest.fit(X_train, y_train)

output = forest.predict(X_test)
print(output)
print(y_test)
count=0
for i in range(len(output)):
    if(output[i]==y_test[i]):
        count=count+1

print(count/len(output))
for i in range(100):
    with open('iris-dtree'+str(i)+'.dot', mode='w') as f:
        tree.export_graphviz(forest.estimators_[i], out_file=f,filled=True,feature_names=X_cols,special_characters=True)


importance=forest.feature_importances_
df_importance = pd.DataFrame(columns=X_cols)
df_importance.loc['importance']=importance

df_importance.to_csv("result.csv")
