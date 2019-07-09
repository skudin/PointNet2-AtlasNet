import os
import shutil
import logging
import sys

import atlasnet2.configuration as conf


def create_folders_for_experiments(experiment_name):
    experiment_path = os.path.join(conf.BASE_PATH, "experiments", experiment_name)
    create_folder_with_dialog(experiment_path)

    snapshots_path = os.path.join(experiment_path, "snapshots")
    os.makedirs(snapshots_path)

    return experiment_path, snapshots_path


def create_folder_with_dialog(path):
    if os.path.exists(path):
        print("%s exists. Remove it? (y/n)" % path)

        answer = str(input()).strip()
        if answer != "y":
            print("Shut down.")
            exit(0)

        shutil.rmtree(path)

    os.makedirs(path)


def set_logging(name, logging_level, logging_to_stdout=False, log_filename=None):
    logger = logging.getLogger(name)
    logger.setLevel(logging_level)

    if logging_to_stdout:
        logger.addHandler(logging.StreamHandler(sys.stdout))

    if log_filename is not None:
        logger.info("Set additional logging to file %s." % log_filename)
        logger.addHandler(logging.FileHandler(log_filename))

    return logger
