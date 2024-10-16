import torch


clue_map_dic = {"name": "name", "should_capi": "should_capi", "unit": "unit", "currency": "currency",
                "stock_percent": "stock_percent"}

def get_entities_result(query, tokenizer, model, label2id):
    """进一步封装识别结果"""
    map_dic = clue_map_dic
    sentence_list, predict_labels = predict(query, tokenizer, model, label2id)
    # print(sentence_list)
    # print(predict_labels)
    if len(predict_labels) == 0:
        # print("句子: {0}\t实体识别结果为空或语句过长".format(query))
        return []
    result = []
    if len(sentence_list) == len(predict_labels):
        result = _bio_data_handler(sentence_list, predict_labels, map_dic)
    res = []
    p = 0
    while p < len(result) and result[p][1] != 'name':
        p += 1
    if p == len(result):
        return [{'name': '', 'should_capi': '', 'unit': '', 'currency' : '', 'stock_percent': ''}]
    while p < len(result):
        j = p+1
        t = {'name': '', 'should_capi': '', 'unit': '', 'currency' : '', 'stock_percent': ''}
        while j < len(result) and result[j][1] != 'name':
            j += 1
        for i in range(p, j):
            if result[i][1] == 'name':
                t['name'] = result[i][0]
            elif result[i][1] == 'should_capi':
                t['should_capi'] = result[i][0].replace("#", "")
            elif result[i][1] == 'unit':
                t['unit'] = result[i][0].replace("#", "")
            elif result[i][1] == 'stock_percent':
                t['stock_percent'] = result[i][0].replace("#", "")
            elif result[i][1] == 'currency':
                t['currency'] = result[i][0].replace("#", "")
        res.append(t)
        p = j
    return res


def predict(sentence, tokenizer, model, label2id):
    """模型预测"""
    # 获取句子的input_ids、token_type_ids、attention_mask
    max_seq_length = 512
    result = tokenizer.encode_plus(sentence)
    input_ids, token_type_ids, attention_mask = result["input_ids"], result["token_type_ids"], result["attention_mask"]
    sentence_list = tokenizer.tokenize(sentence)
    if len(input_ids) > max_seq_length:
        return list(sentence), []
    input_ids = torch.tensor(input_ids, dtype=torch.long)
    token_type_ids = torch.tensor(token_type_ids, dtype=torch.long)
    attention_mask = torch.tensor(attention_mask, dtype=torch.long)

    # 单词在词典中的编码、区分两个句子的编码、指定对哪些词进行self-Attention操作
    input_ids = input_ids.to("cpu").unsqueeze(0)
    token_type_ids = token_type_ids.to("cpu").unsqueeze(0)
    attention_mask = attention_mask.to("cpu").unsqueeze(0)

    # 加载模型
    if isinstance(model, torch.nn.DataParallel):
        model = model.module
    model.eval()
    # 模型预测，不需要反向传播
    with torch.no_grad():
        predict_val = model.predict(input_ids, token_type_ids, attention_mask)

    id2label = {value: key for key, value in label2id.items()}

    predict_labels = []
    for i, label in enumerate(predict_val[0]):
        if i != 0 and i != len(predict_val[0]) - 1:
            predict_labels.append(id2label[label])
    return sentence_list, predict_labels


def _bio_data_handler(sentence, predict_label, map_dic):
    """根据标签序列提取出实体"""
    entities = []
    # 获取初始位置实体标签
    pre_label = predict_label[0]
    # 实体词初始化
    word = ""
    for i in range(len(sentence)):
        # 记录问句当前位置词的实体标签
        current_label = predict_label[i]
        # 若当前位置的实体标签是以B开头的，说明当前位置是实体开始位置
        if current_label.startswith('B'):
            # 当前位置所属标签类别与前一位置所属标签类别不相同且实体词不为空，则说明开始记录新实体，前面的实体需要加到实体结果中
            if pre_label[2:] is not current_label[2:] and word != "":
                entities.append([word, map_dic[pre_label[2:]]])
                # 将当前实体词清空
                word = ""
            # 并将当前的词加入到实体词中
            word += sentence[i]
            # 记录当前位置标签为前一位置标签
            pre_label = current_label
        # 若当前位置的实体标签是以I开头的，说明当前位置是实体中间位置，将当前词加入到实体词中
        elif current_label.startswith('I') or current_label.startswith('M'):
            word += sentence[i]
            pre_label = current_label
        elif current_label.startswith('E'):
            word += sentence[i]
            pre_label = current_label
            if pre_label[2:] is current_label[2:]:
                entities.append([word, map_dic[current_label[2:]]])
                # 将当前实体词清空
                word = ""
        # 若当前位置的实体标签是以O开头的，说明当前位置不是实体，需要将实体词加入到实体结果中
        elif current_label.startswith('O'):
            # 当前位置所属标签类别与前一位置所属标签类别不相同且实体词不为空，则说明开始记录新实体，前面的实体需要加到实体结果中
            if pre_label[2:] is not current_label[2:] and word != "":
                entities.append([word, map_dic[pre_label[2:]]])
            # 记录当前位置标签为前一位置标签
            pre_label = current_label
            # 并将当前的词加入到实体词中
            word = ""
        elif current_label.startswith('S'):
            word += sentence[i]
            pre_label = current_label
    # 收尾工作，遍历问句完成后，若实体刚好处于最末位置，将剩余的实体词加入到实体结果中
    if word != "":
        entities.append([word, map_dic[pre_label[2:]]])
    return entities


if __name__ == '__main__':
    # test_path = os.path.join(os.path.abspath('..'), 'data/test.csv')
    # out_path = os.path.join(os.path.abspath('..'), 'data/output.csv')
    # test_data = pd.read_csv(test_path)
    # # 使用训练好的模型进行预测
    #
    # st_time = time.time()
    #
    # test_data['fmt'] = test_data['context'].apply(get_entities_result)
    #
    # print(time.time() - st_time)
    # test_data.to_csv(out_path)
    # pass

    sen = '李贤平 出资 137.088000万人民币 比例 8.5680%;骆秀苗 出资 145.296000万人民币 比例 9.0810%;黄胜华 出资 137.088000万人民币 比例 8.5680%;李大芃 出资 137.088000万人民币 比例 8.5680%;浙江亿源光电科技有限公司 出资 551.376000万人民币 比例 34.4610%;上海寰杰投资管理有限公司 出资 160.000000万人民币 比例 10.0000%;于松来 出资 332.064000万人民币 比例 20.7540%;'
    print(get_entities_result(sen))