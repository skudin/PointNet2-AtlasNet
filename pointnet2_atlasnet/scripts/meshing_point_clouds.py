import argparse
import os
import os.path as osp
import json
import time
import subprocess

import pointnet2_atlasnet.configuration as conf
import pointnet2_atlasnet.libs.helpers as h


TIMEOUT = 3 * 60


def parse_command_prompt():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="json file of input paths with inference results")
    parser.add_argument("--output", required=True, help="output path")

    return parser.parse_args()


def meshing_point_clouds(input_paths, output_path):
    item_counter = 0
    total_meshing_time = 0.0

    for path in input_paths:
        output_dir = osp.join(output_path, osp.split(osp.normpath(path))[1])
        os.makedirs(output_dir)

        for file_obj in os.listdir(path):
            file_obj_path = osp.join(path, file_obj)

            if not osp.isdir(file_obj_path):
                continue

            try:
                print("Meshing item %s is started." % file_obj)
                start_meshing_time = time.time()

                passed = False
                while not passed:
                    try:
                        ret = subprocess.run(
                            ["python3", "%s/pointnet2_atlasnet/scripts/meshing_point_cloud.py" % conf.BASE_PATH,
                             file_obj_path,
                             osp.join(output_dir, "%s.ply" % file_obj)], timeout=TIMEOUT)

                        if ret.returncode == 0:
                            passed = True
                        else:
                            print("Item was not passed. Retrying.")
                    except subprocess.TimeoutExpired:
                        print("Timeout. Retrying.")

                meshing_time = time.time() - start_meshing_time
                total_meshing_time += meshing_time
                item_counter += 1

                print("Item %s is ready. Meshing time: %f s" % (file_obj, meshing_time))
            except Exception as e:
                print("Item %s is broken." % file_obj)

                raise e

    print("Total items: %d" % item_counter)
    print("Mean meshing time: %f s" % (total_meshing_time / item_counter))


def read_paths(filename):
    with open(filename, "r") as fp:
        return json.load(fp)


def main():
    start_execution_time = time.time()

    args = parse_command_prompt()

    h.create_folder_with_dialog(args.output)
    paths = read_paths(args.input)

    meshing_point_clouds(paths, args.output)

    print("Total execution time: %f s" % (time.time() - start_execution_time))

    print("Done.")


if __name__ == "__main__":
    main()
