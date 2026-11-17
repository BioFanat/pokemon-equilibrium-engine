"""One websocket reader and isolated battle queues for a bounded ladder run.

This module has no Foul Play imports, so routing and admission limits can be
tested without credentials, network access, or the battle engine.
"""
import asyncio
import json
import time
import threading
from pathlib import Path


class BattleMovePool:
    """One independent learner per occupied stream, with a fresh battle context.

    A stream's checkpoint is reused only after its prior battle finishes. The
    factory must give each stream its own state path and each battle its own log.
    """
    def __init__(self, capacity, factory):
        self.capacity, self.factory = capacity, factory
        self.available = set(range(capacity))
        self.battles = {}
        self.lock = threading.Lock()

    def move(self, battle):
        tag = battle.battle_tag
        with self.lock:
            if tag not in self.battles:
                if not self.available:
                    raise RuntimeError("No free learning stream for another battle")
                stream = min(self.available)
                fn = self.factory(tag, stream)
                self.available.remove(stream)
                self.battles[tag] = (stream, fn)
            _, fn = self.battles[tag]
        return fn(battle)

    def stream_for(self, tag):
        with self.lock:
            return self.battles[tag][0] if tag in self.battles else None

    def release(self, tag):
        with self.lock:
            entry = self.battles.pop(tag, None)
            if entry is not None:
                self.available.add(entry[0])


class RoomClient:
    def __init__(self, transport, tag, rating_ready=None):
        self.transport, self.tag = transport, tag
        self.rating_ready = rating_ready
        self.queue = asyncio.Queue()

    async def receive_message(self):
        item = await self.queue.get()
        if isinstance(item, BaseException):
            raise item
        return item

    async def send_message(self, room, messages):
        if room != self.tag:
            raise ValueError("Battle attempted to write into another room")
        # Battle automation does not need to send conversational greetings.
        if messages in (["hf"], ["gg"]):
            return
        await self.transport.send_message(room, messages)

    async def leave_battle(self, tag):
        if tag != self.tag:
            raise ValueError("Battle attempted to leave another room")
        # Ladder ratings can arrive in a separate frame after |win|. The central
        # reader still records these messages while this room waits to leave.
        if self.rating_ready:
            try:
                async with asyncio.timeout(3):
                    while not self.rating_ready(tag):
                        await self.receive_message()
            except TimeoutError:
                pass  # Unrated games need not produce a rating update.
        await self.transport.send_message("", [f"/leave {tag}"])

    async def save_replay(self, tag):
        raise RuntimeError("Public replay uploads are disabled; local replays are recorded automatically")


class ConcurrentLadder:
    def __init__(self, client, *, games, concurrency, format_name, play_battle,
                 completed, failed, should_stop=lambda: False,
                 progress_path=None, search_interval=20.0, search_timeout=600.0,
                 rating_ready=None):
        if not 1 <= concurrency <= 5 or (games is not None and games < 1):
            raise ValueError("Use 1 to 5 concurrent battles and a positive game count")
        self.client = client
        self.games, self.concurrency, self.format = games, concurrency, format_name
        self.play_battle, self.completed, self.failed = play_battle, completed, failed
        self.should_stop, self.progress_path = should_stop, progress_path
        self.rating_ready = rating_ready
        self.search_interval, self.search_timeout = search_interval, search_timeout
        self.rooms, self.tasks, self.retired = {}, {}, set()
        self.started = self.finished = self.failed_count = self.peak_active = 0
        self.pending = False
        self.last_search = float("-inf")
        self.error = None
        self.changed = asyncio.Event()

    def progress(self):
        return dict(target=self.games, concurrency=self.concurrency,
                    started=self.started, completed=self.finished - self.failed_count,
                    failed=self.failed_count, active=list(self.tasks),
                    peak_active=self.peak_active, searching=self.pending,
                    error=self.error, heartbeat_utc=time.time())

    def write_progress(self):
        if self.progress_path:
            path = Path(self.progress_path)
            temporary = path.with_suffix(".tmp")
            temporary.write_text(json.dumps(self.progress(), indent=2) + "\n")
            temporary.replace(path)

    def route(self, message):
        if not message.startswith(">battle-"):
            if message.startswith(("|popup|", "|error|")):
                self.error = "Server refused a command: " + message.split("|", 2)[-1][:500]
                self.changed.set()
            return
        tag = message.split("\n", 1)[0][1:].strip()
        if tag in self.retired:
            return
        if tag not in self.rooms:
            if "|init|battle" not in message:
                return
            if (self.games is not None and self.started >= self.games) or len(self.tasks) >= self.concurrency:
                raise RuntimeError("Server started a battle outside the reserved capacity")
            if not self.pending:
                raise RuntimeError("Unexpected battle without an outstanding ladder search")
            self.pending = False
            self.started += 1
            room = self.rooms[tag] = RoomClient(self.client, tag, self.rating_ready)
            self.tasks[tag] = asyncio.create_task(self._play(tag, room))
            self.peak_active = max(self.peak_active, len(self.tasks))
            self.changed.set()
        self.rooms[tag].queue.put_nowait(message)

    async def _play(self, tag, room):
        try:
            await self.play_battle(room)
            self.completed(tag)
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            self.failed_count += 1
            self.error = f"{tag}: {type(exc).__name__}: {exc}"
            self.failed(tag, exc)
            # A failed agent must not leave an opponent waiting for its timer.
            try:
                await room.send_message(tag, ["/forfeit"])
                await room.leave_battle(tag)
            except Exception:
                pass
        finally:
            self.finished += 1
            self.tasks.pop(tag, None)
            self.rooms.pop(tag, None)
            self.retired.add(tag)
            self.changed.set()

    async def _reader(self):
        try:
            while True:
                self.route(await self.client.receive_message())
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            self.error = f"Connection reader stopped: {type(exc).__name__}: {exc}"
            for room in self.rooms.values():
                room.queue.put_nowait(exc)
            self.changed.set()

    async def run(self):
        reader = asyncio.create_task(self._reader())
        try:
            while True:
                self.changed.clear()
                stopping = self.should_stop() or self.error is not None
                if stopping and self.pending:
                    await self.client.send_message("", ["/cancelsearch"])
                    self.pending = False
                if not self.tasks and not self.pending and (stopping or (self.games is not None and self.started == self.games)):
                    break
                now = time.monotonic()
                if self.pending and now - self.last_search > self.search_timeout:
                    self.error = "No match found within the search timeout"
                    continue
                if (not stopping and not self.pending and (self.games is None or self.started < self.games)
                        and len(self.tasks) < self.concurrency
                        and now - self.last_search >= self.search_interval):
                    # Reserve the slot before sending. Only one search per format
                    # is allowed, even while several battles are running.
                    self.pending = True
                    self.last_search = now
                    try:
                        await self.client.update_team("None")
                        await self.client.search_for_match(self.format)
                    except Exception as exc:
                        self.pending = False
                        self.error = f"Search failed: {type(exc).__name__}: {exc}"
                self.write_progress()
                try:
                    await asyncio.wait_for(self.changed.wait(), timeout=0.5)
                except TimeoutError:
                    pass
            if self.error:
                raise RuntimeError(self.error)
        finally:
            reader.cancel()
            await asyncio.gather(reader, return_exceptions=True)
            for task in list(self.tasks.values()):
                task.cancel()
            await asyncio.gather(*list(self.tasks.values()), return_exceptions=True)
            self.write_progress()
