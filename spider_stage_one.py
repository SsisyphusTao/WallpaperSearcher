import time
import requests
import bs4
from tqdm import tqdm
from threading import Thread
import json


NUM = 500
bar = tqdm(total=NUM)
threads = []
urls = {}

class Worker(Thread):
    def __init__(self, url):
        super().__init__()
        self.url = url

    def run(self):
        try:
            result = requests.get(self.url, timeout=5)
        except:
            time.sleep(1)
            self.run()
            return
        if result.status_code == 429:
            time.sleep(3)
            self.run()
            return
        page_soup = bs4.BeautifulSoup(result.content.decode(),'lxml')
        wallpapaers = [x.get('href') for x in page_soup.select('a') if x.get('class') == ['preview']]
        if not wallpapaers:
            print(self.url)
            print(result.status_code)
        for i in wallpapaers:
            k = i.split('/')[-1]
            urls[k] = i
        bar.update(1)

for i in range(1, NUM+1):
    threads.append(Worker("https://wallhaven.cc/search?categories=110&purity=110&sorting=relevance&order=desc&page=%d"%i))
    threads[-1].start()

for i in threads:
    i.join()

# with open('urls.txt', 'w') as f:
#     for i in urls:
#         f.write(urls[i]+'\n')
with open('urls.json', 'w') as f:
    json.dump(urls, f)