import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fastapi import FastAPI, UploadFile, File, Query, Body
from fastapi.responses import JSONResponse
from datetime import datetime
import asyncio
import shutil

from strategies import strategies
from utils.io import read_ulds, read_packages, load_config

config = load_config(os.path.join(os.path.dirname(__file__), "service.config"))

DEBUG = False


class APIService:
    def __init__(self):
        self.app = FastAPI()
        self.input_folder = config["input folder"]
        self.output_folder = config["output folder"]
        self.default_strategy = strategies[config["default strategy"]]

        if not os.path.exists(self.input_folder):
            os.makedirs(self.input_folder)
        if not os.path.exists(self.output_folder):
            os.makedirs(self.output_folder)

        self.solutions = {}  # Dictionary to store solver instances

        # Register routes
        self.app.post("/filepack")(self.upload_files)
        self.app.get("/status/{request_id}")(self.check_status)
        self.app.get("/solution/{request_id}")(self.get_solution)

    async def _clean_solutions(self):
        """
        Delete the solutions that are older than 24 hours.
        """
        for request_id in self.solutions:
            if (
                datetime.now() - self.solutions[request_id].time_end
            ).total_seconds() > 24 * 60 * 60:
                shutil.rmtree(os.path.join(self.input_folder, request_id))
                del self.solutions[request_id]

    async def upload_files(
        self,
        request_name: str = Body(..., description="Name of the request"),
        k: int = Body(..., description="Number of solutions to generate"),
        uld_file: UploadFile = File(..., description="Uld file"),
        package_file: UploadFile = File(..., description="Package file"),
        strategy: str = Body(
            ...,
            description=f"The strategy to use. Has values: {config['strategies']}",
            required=False,
            default=config["default strategy"],
        ),
    ):
        """
        Upload the files and run the solver.
        """
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            folder_name = f"{request_name}_{timestamp}"
            folder_path = os.path.join(self.input_folder, folder_name)
            output_path = os.path.join(self.output_folder, folder_name)

            if request_name == "":
                return JSONResponse(
                    content={"error": "Request name cannot be empty"}, status_code=400
                )
            elif request_name in self.solutions:
                return JSONResponse(
                    content={"error": "Request name already exists"}, status_code=400
                )

            if k < 0:
                return JSONResponse(
                    content={"error": "K must be greater than or equal to 0"},
                    status_code=400,
                )

            if DEBUG:
                print(f"Saving files to {folder_path}")
                print(f"Uld file: {uld_file}")
                print(f"Package file: {package_file}")
                print(f"Request name: {request_name}")
                print(f"K: {k}")

            os.makedirs(folder_path, exist_ok=True)
            os.makedirs(output_path, exist_ok=True)

            file_paths = {
                "uld": os.path.join(folder_path, "uld.csv"),
                "package": os.path.join(folder_path, "package.csv"),
            }

            with open(file_paths["uld"], "wb") as f:
                shutil.copyfileobj(uld_file.file, f)
            with open(file_paths["package"], "wb") as f:
                shutil.copyfileobj(package_file.file, f)

            ulds = read_ulds(file_paths["uld"])
            packages = read_packages(file_paths["package"])

            if len(ulds) == 0:
                return JSONResponse(
                    content={"error": "Uld file is empty"}, status_code=400
                )
            if len(packages) == 0:
                return JSONResponse(
                    content={"error": "Package file is empty"}, status_code=400
                )

            SelectedStrategy = strategies[strategy]

            self.solutions[request_name] = SelectedStrategy(
                ulds=ulds, packages=packages, k_cost=k
            )

            # Create an async task for the solver using thread pool
            asyncio.create_task(self.solutions[request_name].run())

            return JSONResponse(
                content={
                    "message": "Files uploaded successfully",
                    "request_id": request_name,
                },
                status_code=200,
            )

        except Exception as e:
            return JSONResponse(content={"error": str(e)}, status_code=500)

    async def check_status(self, request_id: str):
        """
        Check the status of the solver.
        """
        try:
            if DEBUG:
                print(f"Checking status for {request_id}")

            if request_id not in self.solutions:
                return JSONResponse(
                    content={"error": "Request ID not found"}, status_code=404
                )

            has_completed = self.solutions[request_id].solution_found
            time_end = (
                self.solutions[request_id].time_end if has_completed else datetime.now()
            )

            return JSONResponse(
                content={
                    "status": ("completed" if has_completed else "solving"),
                    "time_taken": round(
                        time_end - self.solutions[request_id].time_start, 2
                    ),
                }
            )
        except Exception as e:
            return JSONResponse(content={"error": str(e)}, status_code=500)

    async def get_solution(self, request_id: str):
        """
        Get the solution of the solver.
        """
        try:
            if DEBUG:
                print(f"Getting solution for {request_id}")

            if request_id not in self.solutions:
                return JSONResponse(
                    content={"error": "Request ID not found"}, status_code=404
                )

            if not self.solutions[request_id].solution_found:
                return JSONResponse(
                    content={"error": "The solution is not yet available"},
                    status_code=404,
                )

            return JSONResponse(
                content={
                    "request_id": request_id,
                    "solution": await self.solutions[request_id].get_solution_json(),
                }
            )
        except Exception as e:
            return JSONResponse(content={"error": str(e)}, status_code=500)


# Initialize the service
service = APIService()
app = service.app
