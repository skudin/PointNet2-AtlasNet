import os

import atlasnet2.libs.helpers as h
from atlasnet2.libs.settings import Settings
from atlasnet2.libs.network_wrapper import NetworkWrapper


def main():
    settings = Settings("test")

    h.create_folder_with_dialog(os.path.join(settings.experiment_folder, "result", "%d" % settings.snapshot_num))
    pass


if __name__ == "__main__":
    main()
