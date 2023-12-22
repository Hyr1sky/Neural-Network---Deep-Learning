import os
import json
import torch

# 加载词表
def load_vocab():
    word_dict={}
    with open('./Assignment4_dataset/data/vocab.txt') as f:
        for idx,item in enumerate(f.readlines()):
            word_dict[item.strip()]=idx

    return word_dict

# 加载数据
def load_dataset(data_path, is_test):
    examples = []
    with open('./Assignment4_dataset/data/AFQMC数据集/dev.json') as f:
        for line in f.readlines():
            line = json.loads(line)
            text_a = line["sentence1"]
            text_b = line["sentence2"]
            if is_test:
                examples.append((text_a, text_b,))
            else:
                label = line["label"]
                examples.append((text_a, text_b, label))
    return examples

def load_afqmc_data(path):
    train_path=os.path.join(path,'train.json')
    dev_path=os.path.join(path,'dev.json')
    test_path=os.path.join(path,'test.json')

    train_data = load_dataset(train_path, False)
    dev_data = load_dataset(dev_path, False)
    test_data = load_dataset(test_path, True)
    return train_data,dev_data,test_data

# 字符转id
def words2id(example):
    word_dict = load_vocab()
    cls_id = word_dict['[CLS]']
    sep_id = word_dict['[SEP]']

    text_a, text_b, label = example

    input_a = [word_dict[item] if item in word_dict else word_dict['[UNK]'] for item in text_a]
    input_b = [word_dict[item] if item in word_dict else word_dict['[UNK]'] for item in text_b]
    input_ids = [cls_id] + input_a + [sep_id] + input_b + [sep_id]
    segment_id = [0]*(len(input_a)+2) + [1]*(len(input_b)+1)
    return input_ids, segment_id, int(label)

# Dataloader中的collate_fn函数，可参考以下方法，也可以自定义或者使用默认方法
def collate_fn(batch_data, pad_val=0, max_seq_len=512):
    input_ids, segment_ids, labels = [], [], []
    max_len = 0
    for example in batch_data:
        input_id, segment_id, label = example
        # 对数据序列进行截断
        input_ids.append(input_id[:max_seq_len])
        segment_ids.append(segment_id[:max_seq_len])
        labels.append(label)
        # 保存序列最大长度
        max_len = max(max_len, len(input_id))
    # 对数据序列进行填充至最大长度
    for i in range(len(labels)):
        input_ids[i] = input_ids[i]+[pad_val] * (max_len - len(input_ids[i]))
        segment_ids[i] = segment_ids[i]+[pad_val] * (max_len - len(segment_ids[i]))
    return (torch.as_tensor(input_ids), torch.as_tensor(segment_ids),), torch.as_tensor(labels)
