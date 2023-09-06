import logging
import coloredlogs

coloredlogs.install(level=logging.DEBUG, fmt="%(asctime)s %(levelname)s %(message)s")

import robolabel as rl

if __name__ == "__main__":
    app = rl.App()
    app.run()
