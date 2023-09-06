import logging
import coloredlogs

coloredlogs.install(level=logging.DEBUG, fmt="%(asctime)s %(levelname)s %(message)s")

from robolabel.app import App

if __name__ == "__main__":
    app = App()
    app.run()
