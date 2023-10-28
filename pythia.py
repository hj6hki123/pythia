from datasets import load_dataset, Features, Value
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig
from torch.utils.data import DataLoader
from islab.aicup import collate_batch_with_prompt_template, OpenDeidBatchSampler
from transformers import AdamW
from tqdm import trange
from transformers import get_linear_schedule_with_warmup

# 載入數據集
dataset = load_dataset("csv", data_files='./tsv/all.tsv', delimiter='\t',
                       features=Features({
                           'fid': Value('string'), 'idx': Value('int64'),
                           'content': Value('string'), 'label': Value('string')}),
                       column_names=['fid', 'idx', 'content', 'label'], keep_default_na=False)

# 配置模型和 tokenizer
plm = "EleutherAI/pythia-160m-deduped"
bos = '<|endoftext|>'
eos = '<|END|>'
pad = '<|pad|>'
sep ='\n\n####\n\n'
special_tokens_dict = {'eos_token': eos, 'bos_token': bos, 'pad_token': pad, 'sep_token': sep}
tokenizer = AutoTokenizer.from_pretrained(plm, revision="step3000")
tokenizer.padding_side = 'left'
num_added_toks = tokenizer.add_special_tokens(special_tokens_dict)

# 設定訓練參數
EPOCHS = 20
BATCH_SIZE = 8
learning_rate = 1e-4

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
model.to(device)

# 初始化優化器和學習率調度器
optimizer = AdamW(model.parameters(), lr=learning_rate)
scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=EPOCHS)

# 載入數據和設定數據加載器
train_data = list(dataset['train'])
bucket_train_dataloader = DataLoader(train_data, batch_sampler=OpenDeidBatchSampler(train_data, BATCH_SIZE),
                                     collate_fn=lambda batch: collate_batch_with_prompt_template(batch, tokenizer),
                                     pin_memory=True)

# 訓練迴圈
model.train()
for _ in trange(EPOCHS, desc="Epoch"):
    total_loss = 0

    for step, (seqs, labels, masks) in enumerate(bucket_train_dataloader):
        seqs = seqs.to(device)
        labels = labels.to(device)
        masks = masks.to(device)
        model.zero_grad()
        outputs = model(seqs, labels=labels, attention_mask=masks)
        loss = outputs.loss
        loss = loss.mean()

        total_loss += loss.item()
        loss.backward()
        optimizer.step()
        scheduler.step()

    avg_train_loss = total_loss / len(bucket_train_dataloader)
    print("Average train loss: {}".format(avg_train_loss))

# 儲存模型參數
torch.save(model.state_dict(), "temp/allopen160md.pt")
