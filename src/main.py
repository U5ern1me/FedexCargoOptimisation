import os
import sys

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import argparse
from datetime import datetime
import asyncio

from utils.io import read_ulds, read_packages, read_k, write_output, load_config
from strategies import strategies

config = load_config("src/main.config")


async def main():
    # Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--strategy",
        "-s",
        type=str,
        required=False,
        default=config["default strategy"],
        help=f"The strategy to use. Has values: {config['strategies']}",
    )
    args = parser.parse_args()

    SelectedStrategy = strategies[args.strategy]

    # Create output folder if it doesn't exist
    if not os.path.exists(config["output path"]):
        os.makedirs(config["output path"])
    output_path = os.path.join(
        config["output path"],
        f"{args.strategy}-{datetime.now().strftime('%Y-%m-%d_%H-%M')}",
    )

    # Parse input data
    ulds = read_ulds(config["data path"])
    packages = read_packages(config["data path"])
    k = read_k(config["data path"])

    # Run strategy
    strategy = SelectedStrategy(ulds=ulds, packages=packages, k_cost=k)
    await strategy.run()

    # Validate output
    is_valid = await strategy.validate()

    if not is_valid:
        print(f"Invalid solution: {strategy.error}")

    # Save output
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    outputs = await strategy.get_outputs()
    write_output(*outputs, output_path)

    print(
        f"Output saved to {output_path}, time taken: {(strategy.time_end - strategy.time_start):.2f} seconds"
    )


if __name__ == "__main__":
    asyncio.run(main())
