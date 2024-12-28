'''ICC'''
import pandas as pd
from sklearn.metrics import cohen_kappa_score
import warnings
import re
warnings.filterwarnings("ignore")


file1 = r'F:\work\research\chenYouMei\data\金标准第一次评估1000例_v1.xlsx'
file2 = r'F:\work\research\chenYouMei\data\金标准二次评估200例.xlsx'

data1 = pd.read_excel(file1)
data2 = pd.read_excel(file2, sheet_name='Sheet2')

# 去检查号重复的数据
data1.drop_duplicates(subset=['检查号'], keep='first', inplace=True)
data2.drop_duplicates(subset=['检查号'], keep='first', inplace=True)

# 重置index为检查号
data1.index = data1['检查号']
data2.index = data2['检查号']

# 处理CAD-RADS，用re提取数字
data1['CAD-RADS'] = data1['CAD-RADS'].apply(lambda x: re.findall(r'\d+', str(x))[0] if re.findall(r'\d+', str(x)) else None)
data2['CAD-RADS'] = data2['CAD-RADS'].apply(lambda x: re.findall(r'\d+', str(x))[0] if re.findall(r'\d+', str(x)) else None)

choose_columns = ['SIS', 'P分级', 'CAD-RADS', '心肌桥            （主要血管存在=1，不存在=0）', '非钙化斑块         （存在=1，不存在=0）']

for column in choose_columns:
    
    data1_ = data1[column].astype(float)
    data2_ = data2[column].astype(float)

    # denan
    data1_ = data1_.dropna()
    data2_ = data2_.dropna()

    # 求交集
    common_index = data1_.index.intersection(data2_.index)
    data1_ = data1_.loc[common_index]
    data2_ = data2_.loc[common_index]

    kappa = cohen_kappa_score(data1_.values, data2_.values, weights='quadratic')

    print(column, kappa)



