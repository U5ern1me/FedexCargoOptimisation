import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fastapi import FastAPI, UploadFile, File, Query, Body, HTTPException, status
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from datetime import datetime
import asyncio
import shutil
import time
from strategies import strategies
from utils.io import read_ulds, read_packages, load_config, write_output
from concurrent.futures import ThreadPoolExecutor

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
        self.event_loop = asyncio.get_event_loop()

        self.app.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )

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

    async def code_runner(self, request_id: str):
        """
        Code to run the solver.
        """
        try:
            strategy = self.solutions[request_id]["instance"]
            if not await strategy.post_process():
                await strategy.solve()
                await strategy.gravity_stabilization()

                if await strategy.validate():
                    self.solutions[request_id]["status"] = 1
                    write_output(
                        *await strategy.get_outputs(),
                        self.solutions[request_id]["output_path"],
                    )
                else:
                    self.solutions[request_id]["status"] = 2
                    self.solutions[request_id]["error"] = strategy.error
        except Exception as e:
            self.solutions[request_id]["error"] = str(e)
            self.solutions[request_id]["status"] = 2

        self.solutions[request_id]["time_end"] = time.time()

    def code_runner_sync(self, request_id: str):
        asyncio.run(self.code_runner(request_id))

    async def code_runner_wrapper(self, request_id: str):
        executor = ThreadPoolExecutor(max_workers=3)
        await self.event_loop.run_in_executor(
            executor, self.code_runner_sync, request_id
        )

    async def upload_files(
        self,
        request_name: str = Body(..., description="Name of the request"),
        k: int = Body(..., description="Number of solutions to generate"),
        uld_file: UploadFile = File(..., description="Uld file"),
        package_file: UploadFile = File(..., description="Package file"),
        strategy: str = Body(
            ...,
            description=f"The strategy to use. Has values:",
            required=False,
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
                raise HTTPException(
                    detail="Request name cannot be empty",
                    status_code=status.HTTP_400_BAD_REQUEST,
                )
            elif request_name in self.solutions:
                raise HTTPException(
                    detail="Request name already exists",
                    status_code=status.HTTP_400_BAD_REQUEST,
                )

            if k < 0:
                return JSONResponse(
                    detail="K must be greater than or equal to 0",
                    status_code=status.HTTP_400_BAD_REQUEST,
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

            ulds = read_ulds(folder_path)
            packages = read_packages(folder_path)

            if len(ulds) == 0:
                raise HTTPException(
                    detail="Uld file is empty",
                    status_code=status.HTTP_400_BAD_REQUEST,
                )
            if len(packages) == 0:
                raise HTTPException(
                    detail="Package file is empty",
                    status_code=status.HTTP_400_BAD_REQUEST,
                )

            SelectedStrategy = strategies[strategy]

            self.solutions[request_name] = {
                "status": 0,  # 0 - solving, 1 - done, 2 - error
                "time_start": time.time(),
                "time_end": None,
                "error": None,
                "instance": SelectedStrategy(
                    ulds=ulds,
                    packages=packages,
                    k_cost=k,
                    output_path=self.output_folder,
                ),
                "output_path": output_path,
            }

            # Create an async task for the solver using thread pool
            asyncio.create_task(self.code_runner_wrapper(request_name))

            return JSONResponse(
                content={
                    "message": "Files uploaded successfully",
                    "request_id": request_name,
                },
                status_code=status.HTTP_200_OK,
            )
        except HTTPException as e:
            raise e
        except Exception as e:
            raise HTTPException(
                detail=str(e), status_code=status.HTTP_500_INTERNAL_SERVER_ERROR
            )

    async def check_status(self, request_id: str):
        """
        Check the status of the solver.
        """
        try:
            if DEBUG:
                print(f"Checking status for {request_id}")

            if request_id not in self.solutions:
                raise HTTPException(
                    detail="Request ID not found",
                    status_code=status.HTTP_400_BAD_REQUEST,
                )

            run_status = self.solutions[request_id]["status"]

            if run_status == 0:
                content = {
                    "status": "solving",
                    "time_taken": round(
                        time.time() - self.solutions[request_id]["time_start"], 2
                    ),
                }
            elif run_status == 1:
                content = {
                    "status": "completed",
                    "time_taken": round(
                        self.solutions[request_id]["time_end"]
                        - self.solutions[request_id]["time_start"],
                        2,
                    ),
                }
            else:
                raise HTTPException(
                    detail=self.solutions[request_id]["error"],
                    status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                )

            return JSONResponse(content=content)
        except HTTPException as e:
            raise e
        except Exception as e:
            raise HTTPException(
                detail=str(e), status_code=status.HTTP_500_INTERNAL_SERVER_ERROR
            )

    async def get_solution(self, request_id: str):
        """
        Get the solution of the solver.
        """
        try:
            if DEBUG:
                print(f"Getting solution for {request_id}")

            if request_id not in self.solutions:
                raise HTTPException(
                    detail="Request ID not found",
                    status_code=status.HTTP_400_BAD_REQUEST,
                )

            if self.solutions[request_id]["status"] == 0:
                raise HTTPException(
                    detail="The solution is not yet available",
                    status_code=status.HTTP_404_NOT_FOUND,
                )

            if self.solutions[request_id]["status"] == 2:
                raise HTTPException(
                    detail=self.solutions[request_id]["error"],
                    status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                )

            return JSONResponse(
                content={
                    "request_id": request_id,
                    "solution": await self.solutions[request_id][
                        "instance"
                    ].get_solution_json(),
                }
            )
        except HTTPException as e:
            raise e
        except Exception as e:
            raise HTTPException(
                detail=str(e), status_code=status.HTTP_500_INTERNAL_SERVER_ERROR
            )


# Initialize the service
service = APIService()
app = service.app
