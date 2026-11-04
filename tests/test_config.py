"""Credential configuration tests never read or print the real API key."""
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import config


class ConfigTests(unittest.TestCase):
    def test_default_paths_are_inside_the_project(self):
        """Data, database, and optional credentials resolve inside the project, independent of cwd."""
        root = Path(config.__file__).resolve().parent
        for path in [config.DATA_FOLDER, config.DB_PATH, config.SECRETS_FILE]:
            self.assertTrue(Path(path).resolve().is_relative_to(root))

    def test_environment_key_has_priority(self):
        """An environment variable supplies credentials without requiring a secrets file."""
        with patch.dict("os.environ", {"POLYGON_API_KEY":"test-environment-key"}):
            self.assertEqual(config.load_api_key(), "test-environment-key")

    def test_local_file_is_used_when_environment_is_empty(self):
        """The project-local credential file is read only when an environment key is unavailable."""
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder)/"secrets.txt"
            path.write_text("test-file-key\n", encoding="utf-8")
            with patch.dict("os.environ", {"POLYGON_API_KEY":""}), patch.object(config, "SECRETS_FILE", path):
                self.assertEqual(config.load_api_key(), "test-file-key")

    def test_missing_credentials_fail_with_setup_instructions(self):
        """A missing key explains how to configure it instead of referencing another project folder."""
        with tempfile.TemporaryDirectory() as folder:
            with patch.dict("os.environ", {"POLYGON_API_KEY":""}), patch.object(config, "SECRETS_FILE", Path(folder)/"missing.txt"):
                with self.assertRaisesRegex(FileNotFoundError, "Set POLYGON_API_KEY"):
                    config.load_api_key()
