import pandas as pd

path = '../trainData/sample.csv'

deleteWord = ['厦门市','厦门','思明区','思明','厂','店','摊', '有限公司']
df = pd.read_csv(path)

with open('../trainData/temp.csv', 'a', encoding='utf-8') as f:
    for i in range(len(df)):
        name = str(df.loc[i]['公司名称'])
        for x in deleteWord:
            name = name.replace(x,'')
        f.write(name + '\n')




