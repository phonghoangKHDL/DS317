
from pathlib import Path
import os
import tempfile
import time


def write_csv_with_retry(frame, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    data = frame.to_csv(index=False).encode("utf-8")

    handle, name = tempfile.mkstemp(
        prefix=path.name + ".",
        suffix=".tmp",
        dir=path.parent,
    )
    temporary = Path(name)
    try:
        with os.fdopen(handle, "wb") as stream:
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())

        for attempt in range(51):
            try:
                temporary.replace(path)
                return
            except PermissionError as error:
                if attempt == 50:
                    raise PermissionError(
                        f"Windows van khoa {path}. Dong file CSV trong Excel, "
                        "VS Code/Preview, roi chay lai notebook nay. "
                        "Cac fit COMPLETE van duoc giu nguyen."
                    ) from error
                time.sleep(0.1)
    finally:
        try:
            temporary.unlink(missing_ok=True)
        except PermissionError:
            pass


reporting.write_csv = write_csv_with_retry
print("CSV_RETRY_PATCH_ACTIVE: toi da 5 giay, giu cach ghi atomic")
