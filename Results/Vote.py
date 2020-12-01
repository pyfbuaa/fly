import pandas as pd
import json
import os

file_dir = r"G:\program\butterflyDetection\Faster-RCNN-tensorflow-master\Results\VoteResult/"
vote_result = {}

#按93个类别来处理：
for i in range(94):
    df = pd.DataFrame(columns=['img_name', 'confidence', 'xmin', 'ymin', 'xmax', 'ymax'])
    k = 0
    #分别读取三个结果
    for file in os.listdir(file_dir):
        f = open(file_dir + file, 'r', encoding='utf-8')
        result = json.load(f)
        #将每个分类器的第i类检测结果全部收集起来
        for r in result[str(i)]:
            df.loc[k] = r
            k += 1
    #按置信度倒序排列
    df = df.sort_values(by='confidence', ascending=False)
    #取出img_name的重复项
    repeat_df = df[df.img_name.duplicated()]
    #去除重复项，只保留重复的第一项（置信度最高）
    repeat_df.drop_duplicates(subset=['img_name'], keep='first', inplace=True)
    arr = repeat_df.values.tolist()

    vote_result[str(i)] = arr

with open("vote_result.json", 'w') as f:
    json.dump(vote_result, f, ensure_ascii=False)










