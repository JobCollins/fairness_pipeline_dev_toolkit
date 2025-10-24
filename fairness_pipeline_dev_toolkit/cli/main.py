import argparse


def main():
    parser = argparse.ArgumentParser(description="Fairness Toolkit CLI")
    sub = parser.add_subparsers(dest="cmd")
    sub.add_parser("version")
    args = parser.parse_args()
    if args.cmd == "version":
        from fairness_pipeline_dev_toolkit import __version__
        print(__version__)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()