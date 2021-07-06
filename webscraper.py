from urllib.request import urlopen, Request
from bs4 import BeautifulSoup
import pandas as pd

def get_news(ticker):
    finviz_url = 'https://finviz.com/quote.ashx?t='
    ticker = ticker
    url = finviz_url + ticker

    req = Request(url = url, headers = {'user-agent': 'my-app/0.01'})
    response = urlopen(req)

    news_table = {}
    html = BeautifulSoup(response)
    news_table = html.find(id = 'news-table')
    dataRows = news_table.findAll('tr')

    df = pd.DataFrame(columns = ['News_Title', 'Time'])

    for i, table_row in enumerate(dataRows):
        df = df.append({'News_Title': table_row.a.text, 'Time': table_row.td.text}, ignore_index = True)

    return df
