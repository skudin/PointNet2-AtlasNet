import argparse


def parse_command_prompt():
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input", required=True, type=str, help="path to npy dataset")
    parser.add_argument("-o", "--output", required=True, type=str, help="path to writing result")

    args = parser.parse_args()

    return args.input, args.output


def main():
    dataset_path, output_path = parse_command_prompt()
    print("Hello world!")


if __name__ == "__main__":
    main()
