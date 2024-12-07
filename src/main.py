import os
import sys

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import argparse
from datetime import datetime
import asyncio
import logging

from utils.io import read_ulds, read_packages, read_k, write_output, load_config
from utils.visualizer import visualize
from strategies import strategies
from utils.api_error import APIError

config = load_config(os.path.join(os.path.dirname(__file__), "main.config"))


async def main():
    # Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--strategy",
        "-s",
        type=str,
        required=False,
        default=config["default strategy"],
        help=f"The strategy to use. Has values: {', '.join(config['strategies'])}",
    )
    parser.add_argument(
        "--debug",
        "-d",
        action="store_true",
        help="Log debug information",
    )
    parser.add_argument(
        "--visualize",
        "-v",
        action="store_true",
        help="Visualize the solution",
    )
    parser.add_argument(
        "--cache",
        "-c",
        action="store_true",
        help="Do not load cached data",
    )
    args = parser.parse_args()

    # Create output folder if it doesn't exist
    if not os.path.exists(config["output path"]):
        os.makedirs(config["output path"])

    # Create log folder if it doesn't exist
    if not os.path.exists(config["log path"]):
        os.makedirs(config["log path"])

    run_name = f"{args.strategy}-{datetime.now().strftime('%Y-%m-%d_%H-%M')}"
    output_path = os.path.join(config["output path"], run_name)
    log_path = os.path.join(config["log path"], f"{run_name}.log")

    # if debug is true, set environment variable DEBUG to true
    if args.debug:
        print(f"Output logged in {output_path}")
        os.environ["DEBUG"] = "1"
    else:
        os.environ["DEBUG"] = "0"

    if args.cache:
        os.environ["CACHE"] = "0"
    else:
        os.environ["CACHE"] = "1"

    logging.basicConfig(
        level=logging.INFO,
        filename=log_path,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )

    try:
        SelectedStrategy = strategies[args.strategy]
    except KeyError as e:
        if args.strategy == "drl":
            print(
                "Make sure to comment out the DRLStrategy import in src/strategies/__init__.py and install the necessary dependencies"
            )
        elif args.strategy == "gurobi":
            print(
                "Make sure to comment out the GurobiStrategy import in src/strategies/__init__.py and install the necessary dependencies"
            )
        elif args.strategy == "hexaly":
            print(
                "Make sure to comment out the HexalyStrategy import in src/strategies/__init__.py and install the necessary dependencies"
            )
        else:
            if args.debug:
                logging.error(f"Invalid strategy: {args.strategy}")
            print(f"Invalid strategy: {args.strategy}")
        exit()
    except Exception as e:
        if args.debug:
            logging.error(f"Error loading strategy: {e}")
        print(f"Error loading strategy: {e}")
        exit()

    # Parse input data
    ulds = read_ulds(config["data path"])
    packages = read_packages(config["data path"])
    k = read_k(config["data path"])

    # Run strategy
    strategy = SelectedStrategy(
        ulds=ulds, packages=packages, k_cost=k, output_path=config["output path"]
    )
    try:
        await strategy.run()
    except APIError as e:
        logging.error(f"Solver error: {e}")
        print(f"Solver error: {e}")
        print(f"You can check the log file at {log_path} for more information")
        check = input("Do you want to restart using mhpa solver (y/n): ")
        if check == "y":
            if args.debug:
                logging.info("Restarting using mhpa solver")
            strategy.reset()
            os.environ["SOLVER"] = "mhpa"
            await strategy.run()
        else:
            strategy.end()
            exit()
    except Exception as e:
        logging.error(f"Error in strategy: {e}")
        raise e

    print(
        f"Solution calculated in {strategy.time_end - strategy.time_start:.2f} seconds"
    )
    print(f"Validating solution...")

    # Validate output
    is_valid = await strategy.validate()

    if not is_valid:
        print(f"Invalid solution: {strategy.error}")
    else:
        print(f"Total cost: {strategy.calculate_cost()}")
        print(f"Number of packages packed: {strategy.get_num_packed()}")
        print(f"Number of priority ULDs: {strategy.get_num_priority_uld()}")

    # Save output
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    outputs = await strategy.get_outputs()
    write_output(*outputs, output_path)

    print(
        f"Output saved to {output_path}, time taken: {(strategy.time_end - strategy.time_start):.2f} seconds"
    )

    if args.visualize:
        print("Visualizing solution...")
        visualize(output_path, config["data path"])

    # if log file is empty and debug is false, delete logfile
    if not args.debug and os.path.exists(log_path):
        if os.path.getsize(log_path) == 0:
            os.remove(log_path)


if __name__ == "__main__":
    asyncio.run(main())
