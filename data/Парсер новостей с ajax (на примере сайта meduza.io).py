import requests
import json
from bs4 import BeautifulSoup
import csv
from datetime import datetime


def get_html_text(url):
    """
    Функция, возвращающая html страницу
    """
    try:
        res = requests.get(url, timeout=30)
        return res.text

    except requests.ConnectionError:
        print("Connection Error")

    except requests.Timeout:
        print("Timeout Error")

    except requests.RequestException:
        print("Request Exception")


def get_all_news_from_page(html):
    """
    Функция, возвращающая список сслылок на новости на одной странице
    """
    l_links = []
    data = json.loads(html)
    for i in range(0, 24):  # среднее количество статей на странице
        try:
            l_links.append('https://meduza.io/{}'.format(data['collection'][i]))
        except IndexError:
            continue
    return l_links


def get_content(html):
    """
    Функция, возвращающая текст и залоговок новости
    """
    soup = BeautifulSoup(html, 'lxml')
    try:
        text = soup.find('div', 'GeneralMaterial-article').text.strip()
    except AttributeError:
        text = ''
    try:
        title = soup.find('h1').text.strip()
    except AttributeError:
        title = ''
    data = {'text': text, 'title': title}

    return data


def write_csv(data):
    """
    Функция, записывающая данные в csv файл
    """
    with open('meduza_news.csv', 'a', encoding='utf8') as f:
        writer = csv.writer(f)
        writer.writerow((data['text'], data['title']))
        print(data['title'])
        print('--------------------------------')
        print()


def main():
    start = datetime.now()
    for i in range(1, 3200):  # количество страниц на сайте
        print(i)
        url = 'https://meduza.io/api/w5/search?chrono=news&page={}&per_page=24&locale=ru'.format(i)
        all_links = get_all_news_from_page(get_html_text(url))
        for url1 in all_links:
            html = get_html_text(url1)
            data = get_content(html)
            write_csv(data)
        end = datetime.now()
        t = end - start
        print(str(t))


if __name__ == '__main__':
    main()
