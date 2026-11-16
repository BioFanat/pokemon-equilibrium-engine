import asyncio
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'showdown'))
from ladder_concurrency import ConcurrentLadder, RoomClient
import ladder


class Transport:
    def __init__(self):
        self.incoming = asyncio.Queue()
        self.sent = []
        self.searches = 0

    async def receive_message(self):
        return await self.incoming.get()

    async def send_message(self, room, messages):
        self.sent.append((room, messages))

    async def update_team(self, team):
        pass

    async def search_for_match(self, format_name):
        self.searches += 1
        tag = f'battle-test-{self.searches}'
        self.incoming.put_nowait(f'>{tag}\n|init|battle\n|title|Bot vs. Opponent')


class ConcurrentLadderTests(unittest.IsolatedAsyncioTestCase):
    def runner(self, transport, play, completed=lambda tag: None, **kwargs):
        return ConcurrentLadder(transport, games=kwargs.pop('games', 8), concurrency=5,
                                format_name='gen9randombattle', play_battle=play,
                                completed=completed, failed=lambda tag, error: None,
                                search_interval=0, **kwargs)

    async def test_interleaved_messages_are_delivered_only_to_their_room(self):
        transport = Transport()
        gate = asyncio.Event()
        async def play(room):
            await gate.wait()
        runner = self.runner(transport, play)
        for i in (1, 2):
            runner.pending = True
            runner.route(f'>battle-test-{i}\n|init|battle')
        runner.route('>battle-test-2\n|request|second')
        runner.route('>battle-test-1\n|request|first')
        for i, word in [(1, 'first'), (2, 'second')]:
            room = runner.rooms[f'battle-test-{i}']
            self.assertIn('|init|battle', await room.receive_message())
            self.assertEqual(await room.receive_message(), f'>battle-test-{i}\n|request|{word}')
        gate.set()
        await asyncio.gather(*list(runner.tasks.values()))

    async def test_exact_budget_and_five_active_battles(self):
        transport = Transport()
        gate = asyncio.Event()
        completed = []
        async def play(room):
            await room.receive_message()
            await gate.wait()
            await room.send_message(room.tag, ['/choose default'])
        runner = self.runner(transport, play, completed.append, games=13)
        task = asyncio.create_task(runner.run())
        async with asyncio.timeout(3):
            while len(runner.tasks) < 5:
                await asyncio.sleep(0)
            self.assertEqual(transport.searches, 5)
            gate.set()
            await task
        self.assertEqual(runner.peak_active, 5)
        self.assertEqual(len(set(completed)), 13)
        self.assertEqual(runner.started, 13)
        self.assertEqual(transport.searches, 13)

    async def test_room_cannot_send_to_another_battle(self):
        room = RoomClient(Transport(), 'battle-test-1')
        with self.assertRaises(ValueError):
            await room.send_message('battle-test-2', ['/choose move 1'])
        with self.assertRaises(ValueError):
            await room.leave_battle('battle-test-2')

    async def test_worker_error_stops_new_searches_and_forfeits(self):
        transport = Transport()
        async def play(room):
            await room.receive_message()
            raise ValueError('broken move')
        runner = self.runner(transport, play)
        with self.assertRaisesRegex(RuntimeError, 'broken move'):
            await runner.run()
        self.assertEqual(transport.searches, 1)
        self.assertIn(('battle-test-1', ['/forfeit']), transport.sent)
        self.assertEqual(runner.failed_count, 1)

    async def test_stop_drains_active_games_without_new_searches(self):
        transport = Transport()
        stop = False
        async def play(room):
            nonlocal stop
            await room.receive_message()
            stop = True
        runner = self.runner(transport, play, should_stop=lambda: stop)
        await runner.run()
        self.assertEqual(runner.started, 1)
        self.assertEqual(runner.finished, 1)

    async def test_duplicate_room_init_does_not_create_a_second_worker(self):
        transport = Transport()
        gate = asyncio.Event()
        async def play(room):
            await gate.wait()
        runner = self.runner(transport, play)
        runner.pending = True
        msg = '>battle-test-1\n|init|battle'
        runner.route(msg)
        runner.route(msg)
        self.assertEqual(runner.started, 1)
        gate.set()
        await asyncio.gather(*list(runner.tasks.values()))
        runner.route(msg)
        self.assertEqual(runner.started, 1)

    async def test_connection_failure_reaches_waiting_battle(self):
        room = RoomClient(Transport(), 'battle-test-1')
        room.queue.put_nowait(ConnectionError('closed'))
        with self.assertRaises(ConnectionError):
            await room.receive_message()

    async def test_waits_for_delayed_rating_before_leaving(self):
        transport = Transport()
        ready = False
        room = RoomClient(transport, 'battle-test-1', lambda tag: ready)
        task = asyncio.create_task(room.leave_battle(room.tag))
        await asyncio.sleep(0)
        self.assertFalse(task.done())
        self.assertEqual(transport.sent, [])
        ready = True
        room.queue.put_nowait('>battle-test-1\n|raw|Bot rating updated')
        await task
        self.assertEqual(transport.sent, [('', ['/leave battle-test-1'])])

    def test_guards_reject_too_many_games_and_stateful_methods(self):
        args = ladder.parse_args(['--local', '--server', 'ws://127.0.0.1:8000/showdown/websocket',
                                  '--username', 'testbot', '--concurrency', '6'])
        self.assertIn('--concurrency must be from 1 to 5', ladder.refusals(ladder.resolve(args)))
        args.concurrency, args.method = 5, 'hedge_bayes'
        self.assertTrue(any('stateless' in reason for reason in ladder.refusals(ladder.resolve(args))))


if __name__ == '__main__':
    unittest.main()
