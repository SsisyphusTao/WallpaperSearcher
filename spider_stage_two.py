import random
import time
import requests
from multiprocessing import Process
import json

class worker(Process):
    def __init__(self, index) -> None:
        super().__init__()
        self.index=index
        with open('/home/chandler/towhee/lists/%02d.json'%index, 'r') as f:
            self.urls = json.load(f)
        del self.urls['d']
        self.total = len(self.urls)
        self.down = 0

    def catch(self, uid, url):
        try:
            result = requests.get(url, timeout=5)
        except:
            time.sleep(1)
            self.catch(uid, url)
            return
        if result.status_code == 429:
            time.sleep(random.randint(5,10))
            self.catch(uid, url)
            return

        with open('/home/chandler/dataset/html/%s.html'%uid, 'w') as f:
            f.write(result.content.decode())
        # soup = bs4.BeautifulSoup(result.content.decode(),'lxml')
        self.down += 1
        with open('log.log', 'a+') as f:
            print('Process %02d: %03d/%03d'%(self.index, self.total, self.down), file=f)

    def run(self):
        for i in self.urls:
            self.catch(i, self.urls[i])

workers = []
for i in range(48):
    workers.append(worker(i))
    workers[-1].start()

for i in workers:
    i.join()
