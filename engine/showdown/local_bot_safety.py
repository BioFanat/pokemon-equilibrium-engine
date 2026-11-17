"""Boundaries for the human-vs-bot local launcher."""
from urllib.parse import urlsplit


def validate_local_bot_identity(username, password, address):
    target = urlsplit(address)
    if (target.scheme != "ws" or target.hostname != "127.0.0.1"
            or target.username or target.password or target.query or target.fragment
            or target.path != "/showdown/websocket"):
        raise ValueError("Local play bot can connect only to ws://127.0.0.1:<port>/showdown/websocket")
    if username != "eqbot" or password is not None:
        raise ValueError("Local play bot must use the guest nickname eqbot without an account password")
