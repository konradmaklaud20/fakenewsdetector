# -*- coding: utf8 -*-
import requests
from bs4 import BeautifulSoup
import pickle
import keras
import numpy as np
from keras.preprocessing.sequence import pad_sequences
import re
from nltk.corpus import stopwords


total = 0  # Переменная для подчёта найднных соответствий данной новости на различных новостных ресурсах

body = ''  # Тело новости
body = body.lower()
body = re.sub('[^а-яА-я ]', '', body)  # оставляем только слова кириллицы
body = re.sub(r'\s+', ' ', body)  # убираем лишние пробелы

i = ''  # Заголовок новости
i = i.lower()
i = re.sub('[^а-яА-я ]', '', i)
i = re.sub(r'\s+', ' ', i)
print(i)

r1 = requests.get('https://ria.ru/search/?query={}'.format(i)).text
s1 = BeautifulSoup(r1, 'lxml')
try:
    tf = s1.find('div', 'list-item__content').text.strip()  # Проверяем, есть ли новость с исходным заголовком на сайте
    total += 1
except AttributeError:
    pass


r = requests.get('https://www.rbc.ru/search/?project=rbcnews&query={}'.format(i)).text
s = BeautifulSoup(r, 'lxml')
try:
    tf = s.find('span', 'search-item__link-in').text.strip()
    total += 1
except AttributeError:
    pass

# Для поиска на сайте интефакса требуется закодировать поисковый запрос
d = {'а': '%E0',
     'б': '%E1',
     'в': '%E2',
     'г': '%E3',
     'д': '%E4',
     'е': '%E5',
     'ё': '%B8',
     'ж': '%E6',
     'з': '%E7',
     'и': '%E8',
     'й': '%E9',
     'к': '%EA',
     'л': '%EB',
     'м': '%EC',
     'н': '%ED',
     'о': '%EE',
     'п': '%EF',
     'р': '%F0',
     'с': '%F1',
     'т': '%F2',
     'у': '%F3',
     'ф': '%F4',
     'х': '%F5',
     'ц': '%F6',
     'ч': '%F7',
     'ш': '%F8',
     'щ': '%F9',
     'ъ': '%FA',
     'ы': '%FB',
     'ь': '%FC',
     'э': '%FD',
     'ю': '%FE',
     'я': '%FF',
     ' ': '+'}

lst = []
for line in i:
    for character in line:
        if character in d:
            line = line.replace(character, d[character])
    lst.append(line)
a = ''.join(lst)
i2 = a
r2 = requests.get('https://www.interfax.ru/search/?sw={}'.format(i2)).text
s2 = BeautifulSoup(r2, 'lxml')
try:
    tf = s2.find('div', 'sPageResult').text.strip()
    total += 1
except AttributeError:
    pass


totaln = 0  # переменная, аналогичная total, но для поиска на новостных агреготорах

r = requests.get('https://news.google.com/search?q={}&hl=ru&gl=RU&ceid=RU%3Aru'.format(i)).text
s = BeautifulSoup(r, 'lxml')
try:
    tf = s.find_all('article')
    if len(tf) > 2:
        totaln += 1
except AttributeError:
    pass

l1 = []
d1 = {' ': '+'}  # для поиска в news.mail.ru пробелы между словами также необходимо заменить
for line in i:
    for character in line:
        if character in d1:
            line = line.replace(character, d1[character])
    l1.append(line)
a1 = ''.join(l1)
i2 = a1

r = requests.get('https://news.mail.ru/search/?q={}&usid=62'.format(i2)).text
s11 = BeautifulSoup(r, 'lxml')
try:
    tf = s11.find_all('a', 'newsitem__title link-holder')

    tf1 = s11.find_all('a', 'newsitem__title js-preview-list__item js-show_photo link-holder')

    if len(tf) > 1 or len(tf1) > 1:
        totaln += 1
        print('yes', 'mailru')
except AttributeError:
    print('no')

# Затем подключается нейросеть
with open('tokenizer_fake_real_ver2.pickle', 'rb') as handle:
    tokenizer = pickle.load(handle)

new_model = keras.models.load_model('fake_real_model_ver2.h5')
MAX_SEQUENCE_LENGTH = 1075

text = []
text.append(body)

seq = tokenizer.texts_to_sequences(text)
padded = pad_sequences(seq, maxlen=MAX_SEQUENCE_LENGTH)
pred = new_model.predict(padded)
labels = sorted(['satira', 'yellow', 'real', 'fake'])
b = labels[np.argmax(pred)]

if b == 'real' and total == 3 and totaln >= 1:
    ans = 'Это настоящая новость с объективной информацией, написанная качественным журналистским языком'

if b == 'real' and 3 > total > 0 and totaln >= 1:
    ans = 'Похоже, это настоящая новость'

if b == 'real' and total == 0 and totaln <= 1:
    ans = 'Возможно, это новость фейковая, следует проверить её источник'

if b == 'yellow' and total == 3 and totaln >= 1:
    ans = 'Это настоящая новость с более менее объективной информацией'

if b == 'yellow' and 3 > total > 0 and totaln >= 1:
    ans = 'Похоже, это настоящая новость, однако журналист мог быть некомпетентен в некоторых вопросах'

if b == 'yellow' and total == 0 and totaln <= 1:
    ans = 'Возможно, это новость фейковая, взятая из недостоверного источника'

if b == 'fake' and total == 3 and totaln >= 1:
    ans = 'Это настоящая новость, однако в некоторых моментах журналист мог быть предвзятым'

if b == 'fake' and 3 > total > 0 and totaln >= 1:
    ans = 'Похоже, это настоящая новость, однако некоторых моментах журналист мог быть предвзятым'

if b == 'fake' and total == 0 and totaln <= 1:
    ans = 'Возможно, это новость фейковая, взятая из аффелированного источника'

if b == 'satira' and total == 3 and totaln >= 1:
    ans = 'Это настоящая новость, однако язык статьи не является качественным'

if b == 'satira' and 3 > total > 0 and totaln >= 1:
    ans = 'Похоже, это настоящая новость, однако язык статьи не является качественным'

if b == 'satira' and total == 0 and totaln <= 1:
    ans = 'Возможно, это новость фейковая, взятая из источника с плохой репутацией'
