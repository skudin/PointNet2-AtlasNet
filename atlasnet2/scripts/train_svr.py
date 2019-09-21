import atlasnet2.libs.helpers as h
from atlasnet2.libs.settings import Settings
from atlasnet2.libs.svr_wrapper import SVRWrapper


def main():
    settings = Settings("svr", "train")

    experiment_path, snapshots_path, logger, vis = h.init_train(settings)

    network = SVRWrapper(mode="train", dataset_path=settings["dataset"], snapshots_path=snapshots_path,
                         num_epochs=settings["num_epochs"], batch_size=settings["batch_size"],
                         num_workers=settings["num_workers"])
    network.train()


if __name__ == "__main__":
    main()
