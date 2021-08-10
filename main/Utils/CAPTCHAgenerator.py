import logging
logging.getLogger().setLevel(logging.INFO)
import requests
from os import getcwd, listdir
from os.path import join, isfile
from pathlib import Path
import uuid


class CAPTCHAgenerator:

    def __init__(self, num, sourceUrl, name):
        self.sourceUrl = sourceUrl
        self.params = {
            "USER_AGENT": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.164 Safari/537.36"}
        self.num = num
        self.outputPath = getcwd() + "/DataGathering/RawImage/" + name
        Path(self.outputPath).mkdir(parents=True, exist_ok=True)
        self.get_CAPTCHA()

    def get_CAPTCHA(self):
        files = [join(self.outputPath, f) for f in listdir(self.outputPath) if
                 isfile(join(self.outputPath, f)) and ".png" in f]
        if len(files) == self.num:
            logging.warning("Raw images are already Generated")
            return
        if len(files) > self.num:
            dif = len(files) - self.num
            logging.warning("%s files exceed" % str(dif))
            return
        else:
            for i in range(self.num):
                for url in self.sourceUrl:
                    r = requests.get(url, params=self.params)
                    if r.status_code == 200:
                        uuid_str = uuid.uuid4().hex
                        file = open(self.outputPath + "/" +
                                    uuid_str + ".png", "wb")
                        file.write(r.content)
                        file.close()
                        logging.info("Raw images generated")
                    else:
                        logging.warning("Request Error")
