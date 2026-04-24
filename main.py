from dotenv import load_dotenv
load_dotenv()
import os

from .server import app
import uvicorn

def main():
    uvicorn.run(app, host="localhost", port=8001)

main()