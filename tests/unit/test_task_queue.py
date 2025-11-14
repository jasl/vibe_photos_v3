import json
from pathlib import Path

import pytest

from src.services.task_queue import FileTaskQueue


@pytest.fixture()
def queue(tmp_path: Path) -> FileTaskQueue:
    return FileTaskQueue(tmp_path)


def test_enqueue_and_pull_round_trip(queue: FileTaskQueue, tmp_path: Path) -> None:
    image_path = tmp_path / "image.jpg"
    image_path.write_bytes(b"data")

    queue.enqueue(image_path, incremental=False)
    task = queue.pull()
    assert task is not None
    assert task.image_path == image_path
    assert task.incremental is False

    queue.acknowledge(task)
    assert not any(queue.inflight_dir.glob("*.json"))


def test_requeue_moves_file(queue: FileTaskQueue, tmp_path: Path) -> None:
    image_path = tmp_path / "photo.png"
    image_path.write_bytes(b"data")

    queue.enqueue(image_path)
    task = queue.pull()
    assert task is not None

    queue.requeue(task, retry=False)
    failed = list(queue.failed_dir.glob("*.json"))
    assert len(failed) == 1
    with failed[0].open(encoding="utf-8") as handle:
        payload = json.load(handle)
    assert payload["image_path"] == str(image_path)
