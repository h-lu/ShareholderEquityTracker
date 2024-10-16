import os
import pickle

import torch
from transformers import BertTokenizer
import pandas as pd
import time
from tqdm import *

from utils import load_pkl
from sqlalchemy import create_engine
from config import source_params, target_params, source_table_name, target_table_name
from torch_ner.ner_predict import get_entities_result

# 创建数据库连接引擎
source_engine = create_engine(f'mysql+pymysql://{source_params["user"]}:{source_params["password"]}@{source_params["host"]}:{source_params["port"]}/{source_params["db"]}')
target_engine = create_engine(f'mysql+pymysql://{target_params["user"]}:{target_params["password"]}@{target_params["host"]}:{target_params["port"]}/{target_params["db"]}')

query = f"SELECT * FROM `{source_table_name}` where type = '股东股权变更'"
data = pd.read_sql(query, source_engine)

columns = ['eid', 'change_date', 'before_investor', 'before_amount', 'before_unit', 'before_percent',
           'after_investor', 'after_amount', 'after_unit', 'after_percent']
output_data = pd.DataFrame(columns=columns)

def load_model(model_path):
    tokenizer = BertTokenizer.from_pretrained(model_path)
    model = torch.load(os.path.join(model_path, "ner_model.ckpt"), map_location="cpu")
    label2id = load_pkl(os.path.join(model_path, "label2id.pkl"))

    return tokenizer, model, label2id

def find_util(target, fmt):
    for item in fmt:
        if item['name'] == target:
            # 返回item 同时删除item
            fmt.remove(item)
            return item, fmt
    return {'name': '', 'should_capi': '', 'unit': '', 'currency' : '', 'stock_percent': ''}, fmt

def parse_one_row(row, model, tokenizer, label2id):
    eid = row['eid']
    change_date = row['change_date']
    before_context = row['before_content']
    before_fomat = get_entities_result(before_context, tokenizer, model, label2id) if before_context else [{'name': '', 'should_capi': '', 'unit': '', 'currency' : '', 'stock_percent': ''}]
    after_context = row['after_content']
    after_fomat = get_entities_result(after_context, tokenizer, model, label2id) if after_context else [{'name': '', 'should_capi': '', 'unit': '', 'currency' : '', 'stock_percent': ''}]
    for befores in before_fomat:
        before_name = befores['name']
        t,  after_fomat= find_util(before_name, after_fomat)
        output_data.loc[len(output_data)] = [eid, change_date, before_name, befores['should_capi'], befores['unit'] + befores['currency'], befores['stock_percent'],
                                             t['name'], t['should_capi'], t['unit'] + t['currency'], t['stock_percent']]

    if len(after_fomat) > 0:
        for t in after_fomat:
            output_data.loc[len(output_data)] = [eid, change_date, '', '', '', '',
                                             t['name'], t['should_capi'], t['unit'] + t['currency'], t['stock_percent']]


def main():
    model_path = 'model/modelv2_99'
    tokenizer, model ,label2id = load_model(model_path)
    print("============================开始处理====================================")
    for index, row in tqdm(data.iterrows(),total=len(data)):
        parse_one_row(row, model, tokenizer, label2id)

        # if index == 500:
        #     break
    output_data.to_sql(target_table_name, target_engine, index=False, if_exists='append')
    print('数据迁移完成')

    # parse_one_row(data.iloc[77], model, tokenizer, label2id)

if __name__ == '__main__':
    main()