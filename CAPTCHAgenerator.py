import logging
import requests
from os import getcwd, listdir
from os.path import join, isfile
import uuid


class CAPTCHAgenerator:

    def __init__(self, num, dir=None):
        self.source_url = ["http://c.gb688.cn/bzgk/gb/gc?_1627954865562"]
        self.params = {
            "USER_AGENT": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.164 Safari/537.36"}
        self.num = num
        self.CWD = getcwd()
        self.dir = dir if dir else "./RawImage/"
        self.get_CAPTCHA()
    def get_CAPTCHA(self):
        files= [join(self.get_path(), f) for f in listdir(self.get_path()) if isfile(join(self.get_path(), f)) and ".png" in f]
        path =self.get_path()
        if len(files) == self.num:
            logging.debug("Already Generated")
        if len(files) > self.num:
            dif = len(files) - self.num
            logging.debug("%s files exceed" % str(dif))
        else:
            for i in range(self.num):
                for url in self.source_url:
                    r = requests.get(url, params=self.params)
                    if r.status_code == 200:
                        uuid_str = uuid.uuid4().hex
                        file = open(path +
                                    uuid_str + ".png", "wb")
                        file.write(r.content)
                        file.close()
                    else:
                        logging.debug("Request Error")

    def get_path(self):
        return join(self.CWD, self.dir)
