import os
import pandas as pd
import numpy as np
from nltk.corpus import stopwords
import os
import keras
import random
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Dense, Embedding, LSTM, SpatialDropout1D
from sklearn.model_selection import train_test_split
import re
import pymorphy2
from datetime import datetime
seed = 12345
random.seed(seed)

Analyzer = pymorphy2.MorphAnalyzer()


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
Y = pd.get_dummies(df['category']).values


def clean_text(text):
    text = str(text)
    text = text.lower()
    text = re.sub('\n', '', text)
    text = re.sub('[0-9]', '', text)
    text = ' '.join([word for word in text.split() if word not in (stopwords.words('russian'))])
    text = " ".join(Analyzer.parse(word)[0].normal_form for word in text.split())
    return text


df['text'] = df.apply(lambda x: clean_text(x['text']), axis=1)


MAX_NB_WORDS = 50000
MAX_SEQUENCE_LENGTH = 1075
EMBEDDING_DIM = 150

tokenizer = Tokenizer(num_words=MAX_NB_WORDS, filters='!"#$%&()*+,-./:;<=>?@[\]^_`{|}~', lower=True)
tokenizer.fit_on_texts(df['text'].values)
with open('tokenizer_fake_real_ver2.pickle', 'wb') as handle:
    pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)

word_index = tokenizer.word_index


X = tokenizer.texts_to_sequences(df['text'].values)
X = pad_sequences(X, maxlen=MAX_SEQUENCE_LENGTH)


X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.10, random_state=42)
print(X_train.shape, Y_train.shape)
print(X_test.shape, Y_test.shape)

model = Sequential()
model.add(Embedding(MAX_NB_WORDS, EMBEDDING_DIM, input_length=X.shape[1]))
model.add(SpatialDropout1D(0.5))
model.add(LSTM(250, dropout=0.5, recurrent_dropout=0.5))
model.add(Dense(4, activation='softmax'))
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy',
                                                                          keras.metrics.Precision(),
                                                                          keras.metrics.Recall()])

epochs = 3
batch_size = 32

start = datetime.now()

history = model.fit(X_train, Y_train, epochs=epochs, batch_size=batch_size, validation_split=0.1,
                    callbacks=[EarlyStopping(monitor='val_loss', patience=3, min_delta=0.0001)])

end = datetime.now()
t = end - start
print(str(t))
model.save('fake_real_model_ver2.h5')

accr = model.evaluate(X_test, Y_test)
print('Test set\n  Loss: {:0.4f}\n  Accuracy: {:0.4f}'.format(accr[0], accr[1]))
