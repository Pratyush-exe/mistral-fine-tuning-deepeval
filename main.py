from run import run
from modal_config import stub


@stub.local_entrypoint()
def main():
    run.remote()
