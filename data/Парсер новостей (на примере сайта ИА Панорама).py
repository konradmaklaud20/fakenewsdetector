import requests
from bs4 import BeautifulSoup
import csv
from datetime import datetime
from multiprocessing import Pool


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
    soup = BeautifulSoup(html, 'lxml')
    links = soup.find_all('header', class_='entry-header')
    l_links = []
    for i in links:
        a = i.find('a').get('href')
        l_links.append(a)
    return l_links


def get_content(html):
    """
    Функция, возвращающая текст и залоговок новости
    """
    soup = BeautifulSoup(html, 'lxml')
    try:
        text = soup.find('div', class_='entry-content').text.strip()
    except:
        text = ''
    try:
        title = soup.find('h1', class_='entry-title').text.strip()
    except:
        title = ''
    data = {'text': text, 'title': title}

    return data


def pool_func(url):
    """
    Функция для выполнения мультипроцессинга
    """
    html = get_html_text(url)
    data = get_content(html)
    write_csv(data)


def write_csv(data):
    """
    Функция, записывающая данные в csv файл
    """
    with open('panorama_news_satire.csv', 'a', encoding='utf8') as f:
        writer = csv.writer(f)
        writer.writerow((data['text'], data['title']))
        print(data['title'])
        print('--------------------------------')
        print()


def main():
    start = datetime.now()
    for i in range(1, 511):  # итерация по количеству страниц на сайте
        print(i)
        url = 'https://panorama.pub/page/{}?s'.format(i)
        all_links = get_all_news_from_page(get_html_text(url))

        with Pool(20) as pool:
            pool.map(pool_func, all_links)

        end = datetime.now()
        t = end - start
        print(str(t))


if __name__ == '__main__':
    main()
