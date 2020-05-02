import argparse
import os
import os.path as osp


def parse_command_prompt():
    parser = argparse.ArgumentParser()
    parser.add_argument("--reference", required=True, help="path to reference data")
    parser.add_argument("--generated", required=True, help="path to generated data")
    parser.add_argument("--output", required=True, help="output filename")

    return parser.parse_args()


def get_reference_filename(generated_filename, reference_path):
    item_name, _ = osp.splitext(generated_filename)
    reference_file_dir = osp.join(reference_path, item_name)

    for file_obj in os.listdir(reference_file_dir):
        file_obj_path = osp.join(reference_file_dir, file_obj)

        if osp.isfile(file_obj_path) and file_obj.endswith(".ply"):
            return file_obj_path


def get_metro_distance():
    # passed = False
    # while not passed:
    #     try:
    #         ret = subprocess.run(
    #             ["python3", "%s/atlasnet2/scripts/meshing_point_cloud.py" % conf.BASE_PATH, file_obj_path,
    #              osp.join(output_dir, "%s.ply" % file_obj)], timeout=TIMEOUT)
    #
    #         if ret.returncode == 0:
    #             passed = True
    #         else:
    #             print("Item was not passed. Retrying.")
    #     except subprocess.TimeoutExpired:
    #         print("Timeout. Retrying.")
    pass


def get_avg_metro_distance(generated_path, reference_path):
    for file_obj in os.listdir(generated_path):
        generated_filename = osp.join(generated_path, file_obj)

        if not osp.isfile(generated_filename) or not file_obj.endswith(".ply"):
            continue

        reference_filename = get_reference_filename(file_obj, reference_path)

        metro_distance = get_metro_distance()

    pass


def main():
    args = parse_command_prompt()

    avg_metro_distance = get_avg_metro_distance(args.generated, args.reference)

    print("Hello world!")


if __name__ == "__main__":
    main()
