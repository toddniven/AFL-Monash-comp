import requests
from bs4 import BeautifulSoup
import html5lib
import pandas as pd


class Scrape:
    def __init__(self, mapping, proxy):
        self.mapping = mapping
        self.proxy = proxy

    def scrape_history(self, team, table_num):
        mapping = self.mapping
        url = 'https://afltables.com/afl/teams/' + mapping[team] + '/allgames.html'
        r = requests.get(url, headers={'User-Agent': 'test'}, proxies=self.proxy)
        soup = BeautifulSoup(r.text, "html")
        table = soup.find_all('table')[table_num]
        data = []
        rows = table.find_all(['tr', 'th'])
        for row in rows:
            cols = row.find_all('td')
            cols = [ele.text.strip() for ele in cols]
            data.append([ele for ele in cols if ele])
        data = data[14:]
        df = pd.DataFrame(data[1:],
                          columns=['Rnd', 'T', 'Opponent', 'Scoring', 'F', 'Scoring', 'A', 'R', 'M', 'WDL', 'Venue',
                                   'Crowd', 'Date'])
        df = df.drop(df.tail(2).index)
        return df[df['T'] != 'F']

    def scrape_game(self, rnd, year=2019):
        table_num = rnd
        url = 'https://afltables.com/afl/seas/'+str(year)+'.html'
        r = requests.get(url, headers={'User-Agent': 'test'}, proxies=self.proxy)
        soup = BeautifulSoup(r.text, "html")
        table = soup.find_all('table')[table_num]
        data = []
        rows = table.find_all(['tr', 'th'])
        for row in rows:
            cols = row.find_all('td')
            cols = [ele.text.strip() for ele in cols]
            data.append([ele for ele in cols if ele])
        return [data[0][0], data[1][0]]  # returns home team and awasy team.
