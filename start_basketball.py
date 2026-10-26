"""Check the split app/YOLO environments and start the local web app."""

import json
import os
import subprocess
import sys


def _project_path(*parts):
    return os.path.join(os.path.dirname(os.path.abspath(__file__)), *parts)


def _configure_yolo_environment():
    interpreter = _project_path(".venv-yolo", "Scripts", "python.exe")
    worker = _project_path("basketball_yolo_worker.py")
    os.environ.pop("BASKETBALL_YOLO_PYTHON", None)
    if not os.path.isfile(interpreter):
        print(
            "YOLO environment not found; starting with optical-flow tracking only.",
            file=sys.stderr,
        )
        return

    try:
        result = subprocess.run(
            [interpreter, worker, "--check"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=60,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired) as error:
        print(
            "YOLO environment check failed; starting with optical-flow tracking only: "
            "{}".format(error),
            file=sys.stderr,
        )
        return

    output = result.stdout.decode("utf-8", errors="replace").strip().splitlines()
    status = None
    if result.returncode == 0 and output:
        try:
            status = json.loads(output[-1])
        except ValueError:
            status = None

    if not status or status.get("event") != "ready":
        detail = result.stderr.decode("utf-8", errors="replace").strip()
        if not detail:
            detail = "\n".join(output)
        print(
            "YOLO environment check failed; starting with optical-flow tracking only. "
            "{}".format(detail[-1000:]),
            file=sys.stderr,
        )
        return

    os.environ["BASKETBALL_YOLO_PYTHON"] = interpreter
    device = status.get("device", "CUDA" if status.get("cuda_available") else "CPU")
    print("YOLOv8 + ByteTrack ready ({})".format(device))


def main():
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    _configure_yolo_environment()
    from basketball_web import main as start_web_app

    start_web_app()


if __name__ == "__main__":
    main()
