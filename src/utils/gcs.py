import shlex
import subprocess


def gsutil_rsync(source: str, target: str) -> None:
    cmd = f"gsutil rsync -d -r {source} {target}"

    res = subprocess.run(
        shlex.split(cmd),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    if res.returncode != 0:
        print(res)
        raise Exception("gsutil not worked")
