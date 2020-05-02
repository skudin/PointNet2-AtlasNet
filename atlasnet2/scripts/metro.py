import argparse
import os
import os.path as osp
import subprocess
import time
import json
import concurrent.futures as cf

import psutil

TIMEOUT = 60 * 5


def parse_command_prompt():
    parser = argparse.ArgumentParser()
    parser.add_argument("--reference", required=True, help="path to reference data")
    parser.add_argument("--generated", required=True, help="path to generated data")
    parser.add_argument("--generated_type", required=True, choices=("path", "json"), help="type of generated option")
    parser.add_argument("--output", required=True, help="output filename")

    return parser.parse_args()


def get_reference_filename(item_name, reference_path):
    reference_file_dir = osp.join(reference_path, item_name)

    for file_obj in os.listdir(reference_file_dir):
        file_obj_path = osp.join(reference_file_dir, file_obj)

        if osp.isfile(file_obj_path) and file_obj.endswith(".ply"):
            return file_obj_path


def get_metro_distance(item_name, reference_filename, generated_filename):
    start_calculation_time = time.time()

    passed = False
    while not passed:
        try:
            ret = subprocess.run(["/metro/build/metro", reference_filename, generated_filename], timeout=TIMEOUT,
                                 stdout=subprocess.PIPE)

            if ret.returncode == 0:
                passed = True

                stdout = ret.stdout.decode("utf-8")
                pos = stdout.find("Hausdorff")
                value_str = stdout[pos: pos + 40]
                metric_value = float(value_str.split(" ")[2])
            else:
                print("Item was not passed. Retrying.")
        except subprocess.TimeoutExpired:
            print("Timeout. Retrying.")

    calculation_time = time.time() - start_calculation_time

    return item_name, metric_value, calculation_time


def wait_futures(futures):
    item_counter = 0
    avg_metro_distance = 0.0
    total_calculation_time = 0.0

    for future in cf.as_completed(futures):
        error = future.exception()

        if error is not None:
            raise error

        item_name, metric_value, calculation_time = future.result()

        item_counter += 1
        avg_metro_distance += metric_value
        total_calculation_time += calculation_time

        print("Item %s is ready (%d/%d). Metro distance: %f. Calculation metric time: %f s" % (
            item_name, item_counter, len(futures), metric_value, calculation_time))

    avg_metro_distance /= item_counter
    avg_calculation_time = total_calculation_time / item_counter

    return avg_metro_distance, total_calculation_time, avg_calculation_time


def get_avg_metro_distance(generated_path, reference_path):
    with cf.ProcessPoolExecutor(max_workers=psutil.cpu_count(logical=False)) as pool:
        futures = set()
        for file_obj in os.listdir(generated_path):
            generated_filename = osp.join(generated_path, file_obj)

            if not osp.isfile(generated_filename) or not file_obj.endswith(".ply"):
                continue

            item_name, _ = osp.splitext(file_obj)
            reference_filename = get_reference_filename(item_name, reference_path)

            futures.add(pool.submit(get_metro_distance, item_name, reference_filename, generated_filename))

        avg_metro_distance, total_calculation_time, avg_calculation_time = wait_futures(futures)

    print("Total time: %f s" % total_calculation_time)
    print("Avg calculation metric time: %f" % avg_calculation_time)
    print("Avg metro distance: %f" % avg_metro_distance)

    return avg_metro_distance


def write_result(output, metric_value):
    with open(output, "w") as fp:
        json.dump(dict(metro_distance=metric_value), fp=fp, indent=4)


def read_paths(filename):
    with open(filename, "r") as fp:
        return json.load(fp)


def main():
    args = parse_command_prompt()

    if args.generated_type == "path":
        avg_metro_distance = get_avg_metro_distance(args.generated, args.reference)
        write_result(args.output, avg_metro_distance)
    else:
        paths = read_paths(args.generated)

        for path in paths:
            _, task_name = osp.split(path)
            print("Getting average metro distance for %s has started." % task_name)

            avg_metro_distance = get_avg_metro_distance(path, args.reference)
            write_result(osp.join(args.output, "%s.json" % task_name), avg_metro_distance)
        pass

    print("Done.")


if __name__ == "__main__":
    main()
