import sys
from pathlib import Path
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "showdown"))
from local_bot_safety import validate_local_bot_identity


class LocalBotSafetyTests(unittest.TestCase):
    def test_allows_only_local_guest_bot(self):
        validate_local_bot_identity("eqbot", None, "ws://127.0.0.1:8000/showdown/websocket")

    def test_rejects_personal_accounts_and_passwords(self):
        for username, password in [("PersonalAccount", None), ("eqbot", "secret"), ("eqbot", "")]:
            with self.subTest(username=username, password=bool(password)), self.assertRaises(ValueError):
                validate_local_bot_identity(username, password, "ws://127.0.0.1:8000/showdown/websocket")

    def test_rejects_public_servers_and_deceptive_urls(self):
        for address in ["wss://sim3.psim.us/showdown/websocket", "ws://127.0.0.1.example.com/showdown/websocket",
                        "ws://127.0.0.1@sim3.psim.us/showdown/websocket", "ws://user@127.0.0.1/showdown/websocket",
                        "ws://127.0.0.1/showdown/websocket?other=server", "ws://localhost/showdown/websocket"]:
            with self.subTest(address=address), self.assertRaises(ValueError):
                validate_local_bot_identity("eqbot", None, address)
