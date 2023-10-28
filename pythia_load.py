from datasets import load_dataset, Features, Value
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig
from torch.utils.data import DataLoader
from islab.aicup import collate_batch_with_prompt_template, OpenDeidBatchSampler
from transformers import AdamW
from tqdm import trange
from tqdm import tqdm
from islab.aicup import aicup_predict
import io,os

def rename_if_file_exists(file_name):
    if not os.path.exists(file_name):
        return file_name

    file_base, file_ext = os.path.splitext(file_name)
    i = 1
    while os.path.exists(f"{file_base}{i}{file_ext}"):
        i += 1
    new_file_name = f"{file_base}{i}{file_ext}"
    os.rename(file_name, new_file_name)
    print(f"已將 {file_name} 重命名為 {new_file_name}")
    return new_file_name



# 配置模型和 tokenizer
plm = "EleutherAI/pythia-160m-deduped" #"EleutherAI/pythia-70m-deduped"

bos = '<|endoftext|>'
eos = '<|END|>'
pad = '<|pad|>'
sep ='\n\n####\n\n'
special_tokens_dict = {'eos_token': eos, 'bos_token': bos, 'pad_token': pad, 'sep_token': sep}

tokenizer = AutoTokenizer.from_pretrained(plm, revision="step3000",)
tokenizer.padding_side = 'left'
num_added_toks = tokenizer.add_special_tokens(special_tokens_dict)
print(f"{tokenizer.pad_token}: {tokenizer.pad_token_id}")


# 載入模型配置
config = AutoConfig.from_pretrained(plm,
                                    bos_token_id=tokenizer.bos_token_id,
                                    eos_token_id=tokenizer.eos_token_id,
                                    pad_token_id=tokenizer.pad_token_id,
                                    sep_token_id=tokenizer.sep_token_id,
                                    output_hidden_states=False)


# 初始化模型
model = AutoModelForCausalLM.from_pretrained(plm, revision="step3000", config=config)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.resize_token_embeddings(len(tokenizer))



# 載入數據集
from datasets import load_dataset, Features, Value
valid_data = load_dataset("csv", data_files="AICUP/Opendid/opendid_valid.tsv", delimiter='\t',
                          features = Features({
                              'fid': Value('string'), 'idx': Value('int64'),
                              'content': Value('string'), 'label': Value('string')}),
                              column_names=['fid', 'idx', 'content', 'label'])
valid_list= list(valid_data['train'])


model.load_state_dict(torch.load('temp/allopen70m-all.pt'))



BATCH_SIZE = 256


with io.open(rename_if_file_exists('pred_answer/answer.txt'),'w',encoding='utf8') as f:
    for i in tqdm(range(0, len(valid_list), BATCH_SIZE)):
        with torch.no_grad():
            seeds = valid_list[i:i+BATCH_SIZE]
            outputs = aicup_predict(model.to('cuda'), tokenizer, input=seeds)
            for o in outputs:
                f.write(o)
                f.write('\n')