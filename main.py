import logging
import coloredlogs

coloredlogs.install()
logging.basicConfig(format="%(asctime)s %(levelname)s %(message)s", level=logging.DEBUG)

from robolabel.app import App

# change format of logging

if __name__ == "__main__":
    app = App()
    app.run()
