import logging
import coloredlogs

coloredlogs.install(level=logging.INFO, fmt="%(asctime)s %(levelname)s %(message)s")

from robolabel.app import App

# change format of logging

if __name__ == "__main__":
    app = App()
    app.run()
