import atlasnet2.libs.helpers as h
from atlasnet2.libs.settings import Settings
from atlasnet2.libs.svr_wrapper import SVRWrapper


def main():
    settings = Settings("svr", "train")

    experiment_path, snapshots_path, logger, vis = h.init_train(settings)

    network = SVRWrapper()
    network.train()


if __name__ == "__main__":
    main()
