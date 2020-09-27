from transformers import BertTokenizer
from sklearn.model_selection import train_test_split
import torch
import pandas as pd
from transformers import BertForSequenceClassification, AdamW
from transformers import get_linear_schedule_with_warmup
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
import numpy as np
import time
import datetime
import random
import os


seed = 12345
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

base = 'datasets'

panorama = pd.read_csv(os.path.join(base, 'panorama_news_2.csv'))
panorama = panorama.sample(frac=1, random_state=1).reset_index(drop=True)
print(panorama.columns, panorama.shape)

smixer = pd.read_csv(os.path.join(base, 'smixer.csv'))
smixer = smixer.drop(['level_0'], axis=1)
smixer = smixer.sample(frac=1, random_state=1).reset_index(drop=True)
print(smixer.columns, smixer.shape)

vesti_fan = pd.read_csv(os.path.join(base, 'vestifan_news.csv'))
vesti_fan = vesti_fan.drop(['level_0'], axis=1)
vesti_fan = vesti_fan.sample(frac=1, random_state=1).reset_index(drop=True)
print(vesti_fan.columns, vesti_fan.shape)


express = pd.read_csv(os.path.join(base, 'express_gazeta_news.csv'))
express = express.sample(frac=1, random_state=1).reset_index(drop=True)
express = express[:1200]
print(express.columns, express.shape)

katusha = pd.read_csv(os.path.join(base, 'katusha_yellowprop_news.csv'))
katusha = katusha.drop(['level_0'], axis=1)
katusha = katusha.sample(frac=1, random_state=1).reset_index(drop=True)
katusha = katusha[:1200]
print(katusha.columns, katusha.shape)

life = pd.read_csv(os.path.join(base, 'life_news.csv'))
life = life.drop(['Unnamed: 0.1'], axis=1)
life = life.sample(frac=1, random_state=1).reset_index(drop=True)
life = life[:1200]
print(life.columns, life.shape)

rian = pd.read_csv(os.path.join(base, 'rian_yellow_news.csv'))
rian = rian.drop(['level_0'], axis=1)
rian = rian.sample(frac=1, random_state=1).reset_index(drop=True)
rian = rian[:1200]
print(rian.columns, rian.shape)

yoki = pd.read_csv(os.path.join(base, 'yoki_yellow_news.csv'))
yoki = yoki.sample(frac=1, random_state=1).reset_index(drop=True)
yoki = yoki[:1200]
print(yoki.columns, yoki.shape)


echo = pd.read_csv(os.path.join(base, 'echo_moscow_news.csv'))
echo = echo.sample(frac=1, random_state=1).reset_index(drop=True)
echo = echo[:1400]
print(echo.columns, echo.shape)

fontanka = pd.read_csv(os.path.join(base, 'fontanka_news.csv'))
fontanka = fontanka.sample(frac=1, random_state=1).reset_index(drop=True)
fontanka = fontanka[:1400]
print(fontanka.columns, fontanka.shape)

meduza = pd.read_csv(os.path.join(base, 'meduza_news.csv'))
meduza = meduza.sample(frac=1, random_state=1).reset_index(drop=True)
meduza = meduza[:1400]
print(meduza.columns, meduza.shape)

vedomosti = pd.read_csv(os.path.join(base, 'vedomosti_news.csv'))
vedomosti = vedomosti.sample(frac=1, random_state=1).reset_index(drop=True)
vedomosti = vedomosti[:1400]
print(vedomosti.columns, vedomosti.shape)

lenta = pd.read_csv(os.path.join(base, 'lenta-ru-news01012014.csv'))
lenta = pd.DataFrame(
    {'Unnamed: 0': lenta['Unnamed: 0'], 'index': lenta['index'], 'text': lenta['text'], 'title': lenta['title']})
lenta = lenta.sample(frac=1, random_state=1).reset_index(drop=True)
lenta = lenta[:1400]
print(lenta.columns, lenta.shape)


min60 = pd.read_csv(os.path.join(base, '60min_news.csv'))
min60 = min60.drop(['level_0'], axis=1)
min60 = min60.sample(frac=1, random_state=1).reset_index(drop=True)
print(min60.columns, min60.shape)

jp = pd.read_csv(os.path.join(base, 'jp_news.csv'))
jp = jp.sample(frac=1, random_state=1).reset_index(drop=True)
jp = jp[:1100]
print(jp.columns, jp.shape)

nation = pd.read_csv(os.path.join(base, 'nation_news.csv'))
nation = nation.sample(frac=1, random_state=1).reset_index(drop=True)
nation = nation[:1100]
print(nation.columns, nation.shape)

polit = pd.read_csv(os.path.join(base, 'politexpert_news.csv'))
polit = polit.drop(['level_0'], axis=1)
polit = polit.sample(frac=1, random_state=1).reset_index(drop=True)
polit = polit[:1100]
print(polit.columns, polit.shape)

vecher = pd.read_csv(os.path.join(base, 'vecher_news.csv'))
vecher = vecher.sample(frac=1, random_state=1).reset_index(drop=True)
print(vecher.columns, vecher.shape)

fan = pd.read_csv(os.path.join(base, 'fan_news1.csv'))
fan = fan.sample(frac=1, random_state=1).reset_index(drop=True)
fan = fan[:1100]
print(fan.columns, fan.shape)


panorama['category'] = 'satira'
smixer['category'] = 'satira'
vesti_fan['category'] = 'satira'

express['category'] = 'yellow'
katusha['category'] = 'yellow'
life['category'] = 'yellow'
rian['category'] = 'yellow'
yoki['category'] = 'yellow'

echo['category'] = 'real'
fontanka['category'] = 'real'
meduza['category'] = 'real'
vedomosti['category'] = 'real'
lenta['category'] = 'real'

min60['category'] = 'fake'
jp['category'] = 'fake'
nation['category'] = 'fake'
polit['category'] = 'fake'
vecher['category'] = 'fake'
fan['category'] = 'fake'

df = pd.concat([panorama, smixer, vesti_fan,
                express, katusha, life, rian, yoki,
                lenta, echo, fontanka, meduza, vedomosti,
                fan, min60, jp, nation, polit, vecher])


print(df.shape)

df = df.dropna()
df = df.sample(frac=1, random_state=1).reset_index(drop=True)



# df['text'] = df['text'].str[:1200] в качестве эксперимента можно попробовать уменьшить размер текстового поля

df_train, df_test = train_test_split(df, test_size=0.15, random_state=2020)

train_labels = df_train['category'].tolist()
train_text = df_train['text'].tolist()

test_labels = df_test['category'].tolist()
test_text = df_test['text'].tolist()


def format_label_value(label_list):
    """
    Функция, которая переводит числа в диапазон, начиная с нуля,
    поскольку этого требует BERT
    """
    format_label_list = []

    for label in label_list:
        format_label_list.append(label - 1)

    return format_label_list


train_labels = format_label_value(train_labels)
test_labels = format_label_value(test_labels)

# определяем гиперпараметры
pretrained_model = 'bert-base-multilingual-cased'
batch_size = 32
lr = 5e-5
eps = 1e-8
epochs = 2
num_warmup_steps = 0


tokenizer = BertTokenizer.from_pretrained(pretrained_model, do_lower_case=True)


def text_to_id(tokenizer_in_func, text_list):
    """
    Функция для токенизации текста и
    добавления в него токенов [CLS] и [SEP] для подачи в BERT
    """
    ids_list = []
    for item in text_list:
        encoded_item = tokenizer_in_func.encode(item, add_special_tokens=True)
        ids_list.append(encoded_item)

    return ids_list


train_text_ids = text_to_id(tokenizer, train_text)
test_text_ids = text_to_id(tokenizer, test_text)


def padding_truncating(input_ids_list, max_length):
    """
    Функция для добавления паддинга
    Приводит все предложения к одной длине
    """
    processed_input_ids_list = []
    for item in input_ids_list:

        if len(item) < max_length:
            seq_list = [0] * (max_length - len(item))
            item = item + seq_list

        elif len(item) >= max_length:
            item = item[:max_length]

        processed_input_ids_list.append(item)

    return processed_input_ids_list


train_padding_list = padding_truncating(train_text_ids, max_length=50)
test_padding_list = padding_truncating(test_text_ids, max_length=50)


def get_attention_masks(pad_input_ids_list):
    """
    Функция, которая создаёт маску для механизма внимания
    """
    attention_masks_list = []

    for item in pad_input_ids_list:

        mask_list = []
        for subitem in item:
            if subitem > 0:
                mask_list.append(1)
            else:
                mask_list.append(0)
        attention_masks_list.append(mask_list)

    return attention_masks_list


train_attention_masks = get_attention_masks(train_padding_list)
test_attention_masks = get_attention_masks(test_padding_list)

train_padding_list, validation_padding_list, \
    train_labels, validation_labels, \
    train_attention_masks, validation_attention_masks = train_test_split(train_padding_list,
                                                                         train_labels,
                                                                         train_attention_masks,
                                                                         random_state=2020,
                                                                         test_size=0.1)


train_inputs = torch.tensor(train_padding_list)
validation_inputs = torch.tensor(validation_padding_list)
test_inputs = torch.tensor(test_padding_list)

train_labels = torch.tensor(train_labels)
validation_labels = torch.tensor(validation_labels)
test_labels = torch.tensor(test_labels)

train_masks = torch.tensor(train_attention_masks)
validation_masks = torch.tensor(validation_attention_masks)
test_masks = torch.tensor(test_attention_masks)

train_data = TensorDataset(train_inputs, train_masks, train_labels)
train_sampler = RandomSampler(train_data)
train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=batch_size)

validation_data = TensorDataset(validation_inputs, validation_masks, validation_labels)
validation_sampler = RandomSampler(validation_data)
validation_dataloader = DataLoader(validation_data, sampler=validation_sampler, batch_size=batch_size)

test_data = TensorDataset(test_inputs, test_masks, test_labels)
test_sampler = SequentialSampler(test_data)
test_dataloader = DataLoader(test_data, sampler=test_sampler, batch_size=batch_size)

model = BertForSequenceClassification.from_pretrained(
    pretrained_model, num_labels=4, output_attentions=False, output_hidden_states=False)


model.to(device)
optimizer = AdamW(model.parameters(),
                  lr=lr,
                  eps=eps)

total_steps = len(train_dataloader) * epochs

scheduler = get_linear_schedule_with_warmup(optimizer,
                                            num_warmup_steps=num_warmup_steps,
                                            num_training_steps=total_steps)


def flat_accuracy(preds, labels):
    """
    Функция для подсчёта accuracy
    """
    pred_flat = np.argmax(preds, axis=1).flatten()
    labels_flat = labels.flatten()
    return np.sum(pred_flat == labels_flat) / len(labels_flat)


def format_time(spent_time_function):
    """
    Функция для замера времени
    """
    elapsed_rounded = int(round(spent_time_function))
    return str(datetime.timedelta(seconds=elapsed_rounded))


for epoch_i in range(epochs):
    print('Start')
    t0 = time.time()
    total_loss = 0
    model.train()

    for step, batch in enumerate(train_dataloader):

        if step % 10 == 0 and not step == 0:
            elapsed = format_time(time.time() - t0)
            print('Batch {:>5,} of {:>5,}. Elapsed: {:}.'.format(step, len(train_dataloader), elapsed))

        b_input_ids = batch[0].to(device)
        b_input_mask = batch[1].to(device)
        b_labels = batch[2].to(device)

        model.zero_grad()

        outputs = model(b_input_ids,
                        token_type_ids=None,
                        attention_mask=b_input_mask,
                        labels=b_labels)

        loss = outputs[0]
        total_loss += loss.item()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()

    avg_train_loss = total_loss / len(train_dataloader)

    t0 = time.time()
    model.eval()
    eval_loss, eval_accuracy = 0, 0
    nb_eval_steps, nb_eval_examples = 0, 0

    for batch in validation_dataloader:
        batch = tuple(t.to(device) for t in batch)
        b_input_ids, b_input_mask, b_labels = batch
        with torch.no_grad():
            outputs = model(b_input_ids,
                            token_type_ids=None,
                            attention_mask=b_input_mask)
        logits = outputs[0]
        logits = logits.detach().cpu().numpy()
        label_ids = b_labels.to('cpu').numpy()
        tmp_eval_accuracy = flat_accuracy(logits, label_ids)
        eval_accuracy += tmp_eval_accuracy
        nb_eval_steps += 1

    print("Accuracy: {0:.2f}".format(eval_accuracy / nb_eval_steps))

print('Finish')

saved_model_dir = "./saved_models/"

if not os.path.exists(saved_model_dir):
    os.makedirs(saved_model_dir)

model.save_pretrained(saved_model_dir)
tokenizer.save_pretrained(saved_model_dir)
