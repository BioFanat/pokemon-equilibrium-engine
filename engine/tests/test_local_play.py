"""Local-play bootstrap boundaries; no downloads or real servers required."""
import hashlib
import io
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import tarfile
import tempfile
import types
import unittest
from unittest.mock import Mock, patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "showdown"))
import local_play as lp


class LocalPlayTests(unittest.TestCase):
    def test_checksums_accept_exact_bytes_and_reject_changes(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "download"
            p.write_bytes(b"trusted bytes")
            lp.verify_sha256(p, hashlib.sha256(b"trusted bytes").hexdigest())
            with self.assertRaises(lp.SetupError):
                lp.verify_sha256(p, "0" * 64)

    def test_archive_rejects_traversal_and_outside_links(self):
        with tempfile.TemporaryDirectory() as tmp:
            for member in ("../escape", "/absolute-escape", "link"):
                archive = Path(tmp) / "bad.tar"
                with tarfile.open(archive, "w") as out:
                    info = tarfile.TarInfo(member)
                    if member == "link":
                        info.type = tarfile.SYMTYPE
                        info.linkname = "../../escape"
                    out.addfile(info)
                with self.assertRaises(lp.SetupError):
                    lp.extract_archive(archive, Path(tmp) / "install")

    def test_archive_keeps_safe_relative_links(self):
        with tempfile.TemporaryDirectory() as tmp:
            archive = Path(tmp) / "node.tar"
            with tarfile.open(archive, "w") as out:
                body = b"node"
                regular = tarfile.TarInfo("node/bin/node")
                regular.size = len(body)
                out.addfile(regular, io.BytesIO(body))
                link = tarfile.TarInfo("node/bin/node-link")
                link.type = tarfile.SYMTYPE
                link.linkname = "node"
                out.addfile(link)
            lp.extract_archive(archive, Path(tmp) / "install")
            self.assertEqual((Path(tmp) / "install/node/bin/node-link").read_bytes(), b"node")

    def test_server_config_always_binds_loopback(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "config").mkdir()
            (root / "server").mkdir()
            (root / "server/ip-tools.ts").write_text("void IPTools.updateTorRanges();\n")
            (root / "config/config-example.js").write_text("exports.bindaddress = '0.0.0.0';\n")
            lp.configure_server(root)
            config = (root / "config/config.js").read_text()
            self.assertTrue(config.rfind("'127.0.0.1'") > config.rfind("'0.0.0.0'"))
            self.assertIn("exports.repl = false", config)
            self.assertNotIn("void IPTools.updateTorRanges();", (root / "server/ip-tools.ts").read_text())
            lp.configure_server(root)  # idempotent scoped patch

    def test_unrecognized_dependency_is_never_overwritten(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "foul-play").mkdir()
            sentinel = root / "foul-play/my-file"
            sentinel.write_text("keep me")
            with self.assertRaises(lp.SetupError):
                lp.ensure_source(root, "foul-play")
            self.assertEqual(sentinel.read_text(), "keep me")

    def test_runtime_environment_does_not_modify_shell_configuration(self):
        original = dict(os.environ)
        env = lp.tool_environment(Path("/tmp/example-local-play"))
        self.assertEqual(dict(os.environ), original)
        self.assertEqual(env["CARGO_HOME"], "/tmp/example-local-play/tools/cargo")
        self.assertEqual(env["OMP_NUM_THREADS"], "1")

    def test_existing_port_never_starts_or_stops_a_process(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "environment.json").write_text("{}")
            with patch.object(lp, "port_is_open", return_value=True), patch.object(lp.subprocess, "Popen") as spawn:
                with self.assertRaises(lp.SetupError):
                    lp.play(root, types.SimpleNamespace(port=8000))
                spawn.assert_not_called()

    def test_cleanup_terminates_only_owned_process_group(self):
        process = Mock(pid=12345)
        with patch.object(lp.os, "killpg") as kill:
            lp.stop_process(process)
            kill.assert_called_once_with(12345, signal.SIGTERM)

    def test_cleanup_escalates_a_stuck_owned_process(self):
        process = Mock(pid=12345)
        process.wait.side_effect = [subprocess.TimeoutExpired("bot", 8), 0]
        with patch.object(lp.os, "killpg") as kill:
            lp.stop_process(process)
            self.assertEqual([call.args for call in kill.call_args_list],
                             [(12345, signal.SIGTERM), (12345, signal.SIGKILL)])

    def test_node_version_detection_does_not_accept_node20(self):
        with patch.object(lp.subprocess, "check_output", return_value="v20.13.1\n"):
            self.assertEqual(lp.node_major("node"), 20)


if __name__ == "__main__":
    unittest.main()
