import json
import pandas as pd
import numpy as np

# 加载理论得分
theory_scores = {}
for country in ['France', 'Germany', 'Italy']:
    with open(f'cache/theory_scores_{country}.json', 'r', encoding='utf-8') as f:
        data = json.load(f)
        theory_scores[country] = np.array([
            data['final']['theory_scores']['x_market'],
            data['final']['theory_scores']['x_political'],
            data['final']['theory_scores']['x_institutional']
        ])

# 加载投票数据
df = pd.read_excel('real_vote.xlsx')

print('理论得分:')
for country, scores in theory_scores.items():
    print(f'{country}: {scores}')

print('\n投票数据:')
for country in ['France', 'Germany', 'Italy']:
    rows = df[(df.iloc[:, 0].astype(str).str.contains(country, case=False, na=False)) | 
              (df.iloc[:, 1].astype(str).str.contains(country, case=False, na=False))]
    if not rows.empty:
        print(f'{country}: 第一次={rows.iloc[0, 2]}, 第二次={rows.iloc[0, 3]}')

print('\n所有投票数据:')
for idx, row in df.iterrows():
    print(f"行{idx}: {row.iloc[0]} / {row.iloc[1]} - 第一次={row.iloc[2]}, 第二次={row.iloc[3]}")
