# For fix SIGSEGV.
import open3d

import pointnet2_atlasnet.libs.helpers as h
from pointnet2_atlasnet.libs.settings import Settings
from pointnet2_atlasnet.libs.network_wrapper import NetworkWrapper


def main():
    settings = Settings("train")

    experiment_path, snapshots_path, logger, vis = h.init_train(settings)

    network = NetworkWrapper(svr=True, mode="train", vis=vis, dataset_path=settings["dataset"],
                             snapshots_path=snapshots_path,
                             num_epochs=settings["num_epochs"], batch_size=settings["batch_size"],
                             num_workers=settings["num_workers"], encoder_type=settings["encoder_type"],
                             pretrained_ae=settings["pretrained_ae"], num_points=settings["num_points"],
                             num_primitives=settings["num_primitives"], bottleneck_size=settings["bottleneck_size"],
                             learning_rate=settings["learning_rate"],
                             epoch_num_reset_optimizer=settings["epoch_num_reset_optimizer"],
                             multiplier_learning_rate=settings["multiplier_learning_rate"])
    network.train()


if __name__ == "__main__":
    main()
