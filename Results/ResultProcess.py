import pandas as pd
import json

file_dir = r"./Results/result.csv"

def top_process(data, n):
    data.sort_values(by='confidence', ascending=False, inplace=True)
    data_grouped = data.groupby("img_name").head(n)

    return data_grouped

def thresh_process(data, thresh):

    return data[data['confidence']>=thresh]


if __name__ =='__main__':

    result = pd.read_csv(file_dir, header=None,
                         names=['img_name', 'class_label', 'confidence', 'xmin', 'ymin', 'xmax', 'ymax'])

    #每张图只留置信度topN的预测结果
    result = top_process(result, 2)

    #每张图只留置信度大于阈值的预测结果
    # result = thresh_process(result, 0.5)

    # 把数据处理成提交格式的json
    result_json = {}
    grouped = result.groupby("class_label")
    for value, group in grouped:
        # print(group[['img_name', 'confidence', 'xmin', 'ymin', 'xmax', 'ymax']])
        result_json[str(value)] = group[['img_name', 'confidence', 'xmin', 'ymin', 'xmax', 'ymax']].values.tolist()
    with open("./result.json", 'w') as f:
        json.dump(result_json, f, ensure_ascii=False)
        print("数据写入json文件完成...")





