import pandas as pd

path = '../trainData/sample.csv'

df = pd.read_csv(path)

with open('../trainData/temp.csv', 'a', encoding='utf-8') as f:
    for i in range(len(df)):
        name = str(df.loc[i]['公司名称'])
        newName = name.replace('厦门市', '').replace('厦门', '').replace('思明区', '').replace('思明', '').replace('厂', '').replace('店', '').replace('摊', '').replace('有限公司', '')
        f.write(newName + '\n')




