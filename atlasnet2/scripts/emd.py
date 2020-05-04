import argparse


def parse_command_prompt():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="path to inference results")
    parser.add_argument("--output", required=True, help="output filename")

    return parser.parse_args()


def main():
    args = parse_command_prompt()

    print("Hello world!")


if __name__ == "__main__":
    main()