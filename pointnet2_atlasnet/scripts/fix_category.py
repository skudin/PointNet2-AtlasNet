import os
import os.path as osp
import json


def main():
    path = "/app/data/36_wax_ups/train/"

    new_meta = dict(category="wax_up")

    for file_obj in os.listdir(path):
        file_obj_path = osp.join(path, file_obj)

        if not osp.isdir(file_obj_path):
            print("Case %s is skipped." % file_obj)

            continue

        with open(osp.join(file_obj_path, "meta.json"), "w") as fp:
            json.dump(new_meta, fp=fp, indent=4)

        print("Case %s is processed." % file_obj)

    print("Done.")


if __name__ == "__main__":
    main()
