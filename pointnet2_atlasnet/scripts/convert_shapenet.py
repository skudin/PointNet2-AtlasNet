import argparse
import os
import shutil
import json
import concurrent.futures as cf
from typing import Tuple
from collections import namedtuple

import numpy as np
from plyfile import PlyData


def parse_command_prompt() -> Tuple[str, str]:
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input", required=True, type=str, help="path to raw dataset")
    parser.add_argument("-o", "--output", required=True, type=str, help="path for writing results")

    args = parser.parse_args()

    return args.input, args.output


def make_index(dataset_path: str):
    img_path = os.path.join(dataset_path, "ShapeNet", "ShapeNetRendering")
    point_clouds_path = os.path.join(dataset_path, "customShapeNet")

    categories = get_categories(dataset_path)

    train_index = list()
    test_index = list()

    for category in categories:
        img_folder = os.path.join(img_path, categories[category])
        img_folder_index = sorted(os.listdir(img_folder))

        ply_folder = os.path.join(point_clouds_path, categories[category], "ply")

        # noinspection PyBroadException
        try:
            ply_folder_index = sorted(os.listdir(ply_folder))
        except:
            ply_folder_index = []

        index = [value for value in img_folder_index if value + ".points.ply" in ply_folder_index]
        print("Category: %s, amount of files: %d, %f %%" % (category, len(index), len(index) / len(img_folder_index)))

        Case = namedtuple("Case", "id category ply_model renders")

        meta = list()
        for item in index:
            ply_model = os.path.join(ply_folder, item + ".points.ply")
            renders_path = os.path.join(img_folder, item, "rendering")
            renders = sorted(
                [os.path.join(renders_path, item) for item in os.listdir(renders_path) if item.endswith(".png")])

            meta.append(Case(item, category, ply_model, renders))

        train_index.extend(meta[: int(0.8 * len(meta))])
        test_index.extend(meta[int(0.8 * len(meta)):])

    return train_index, test_index, list(categories.keys())


def get_categories(dataset_path: str):
    result = dict()

    with open(os.path.join(dataset_path, "synsetoffset2category.txt"), "r") as fp:
        for line in fp:
            tokens = line.strip().split()
            result[tokens[0]] = tokens[1]

    return result


def convert(train_index, test_index, categories, output_path):
    with open(os.path.join(output_path, "categories.json"), "w") as fp:
        json.dump(categories, fp)

    train_part_path = os.path.join(output_path, "train")
    os.mkdir(train_part_path)

    test_part_path = os.path.join(output_path, "test")
    os.mkdir(test_part_path)

    train_part_error_counter = convert_dataset_part(train_index, train_part_path)
    test_part_error_counter = convert_dataset_part(test_index, test_part_path)

    print("Train part: processing of %d cases out of %d failed." % (train_part_error_counter, len(train_index)))
    print("Test part: processing of %d cases out of %d failed." % (test_part_error_counter, len(test_index)))


def convert_dataset_part(index, output_path):
    futures = set()

    with cf.ProcessPoolExecutor() as executor:
        for case in index:
            future = executor.submit(process_case, case.id, case.category, case.ply_model, case.renders, output_path)
            futures.add(future)

    error_counter = wait_futures(futures)

    return error_counter


def wait_futures(futures):
    error_counter = 0

    for future in cf.as_completed(futures):
        result = future.result()
        if not result:
            error_counter += 1

    return error_counter


def process_case(case_id, category, ply_model, renders, output_path):
    case_name = "%s_%s" % (category, case_id)
    print("Conversion of case %s started." % case_name)

    # Reading ply model.
    try:
        with open(ply_model, "rb") as fp:
            ply_data = PlyData.read(fp)
    except Exception as e:
        print("Case %s exception during parsing: %s" % (case_name, str(e)))
        print("Model file is not correct.")

        return False

    case_path = os.path.join(output_path, case_name)
    os.mkdir(case_path)

    renders_path = os.path.join(case_path, "renders")
    os.mkdir(renders_path)

    # Writing meta information of case.
    with open(os.path.join(case_path, "meta.json"), "w") as fp:
        json.dump({"category": category}, fp)

    # Copying renders.
    for image in renders:
        shutil.copy(image, os.path.join(renders_path, os.path.basename(image)))

    # Converting ply model.
    point_cloud = np.vstack([ply_data["vertex"][key].T for key, _ in ply_data["vertex"].data.dtype.descr]).T
    np.save(os.path.join(case_path, "point_cloud.npy"), point_cloud)

    print("Case %s converted." % case_name)

    return True


def main():
    input_path, output_path = parse_command_prompt()

    if os.path.exists(output_path):
        shutil.rmtree(output_path)
    os.makedirs(output_path)

    train_index, test_index, categories = make_index(input_path)
    convert(train_index, test_index, categories, output_path)

    print("Done!")


if __name__ == "__main__":
    main()
