from run import run
from common import stub


@stub.local_entrypoint()
def main():
    run.remote()
