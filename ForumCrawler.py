import requests
import json
import pickle
from bs4 import BeautifulSoup


class scrape_forum():
    def __init__(self, url):
        self.url_prefix = url
        self.topics = []

    def crawler(self, maxPage=200):
        numPage = 1
        while (True):
            response = requests.get(self.url_prefix + str(numPage)).text
            content = json.loads(response)
            if (len(content['topic_list']['topics']) == 0
                    or numPage > maxPage):
                break
            for topic in content['topic_list']['topics']:
                self.topics.append(topic["title"])
            numPage += 1

    def parse_data(self, url):
        headers = {
            'User-Agent':
            'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/83.0.4103.97 Safari/537.36'
        }
        response = requests.get(url, headers=headers).text
        soup = BeautifulSoup(response, 'lxml')
        topic_list = soup.find_all('span', class_='link-top-line')
        for topic in topic_list:
            self.topics.append(topic.get_text().strip())


if __name__ == "__main__":
    Hopscotch_url = "https://forum.gethopscotch.com/latest"
    Hopscotch_urlPrefix = "https://forum.gethopscotch.com/latest.json?ascending=false&no_definitions=true&order=default&page="
    Hopscotch = scrape_forum(Hopscotch_urlPrefix)
    Hopscotch.parse_data(Hopscotch_url)
    Hopscotch.crawler()
    print(len(Hopscotch.topics))

    with open("Hopscotch_url_topics.pickle", "wb") as output:
        pickle.dump(Hopscotch.topics, output)