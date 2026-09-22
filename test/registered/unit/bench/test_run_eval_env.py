from unittest.mock import patch

from sglang.test.run_eval import _local_eval_subprocess_env
from sglang.test.test_utils import CustomTestCase


class TestRunEvalEnvironment(CustomTestCase):
    def test_local_hosts_are_appended_to_both_no_proxy_variants(self):
        with patch.dict(
            "os.environ",
            {
                "HTTP_PROXY": "http://proxy.invalid:8080",
                "NO_PROXY": "existing.internal,localhost",
                "no_proxy": "existing.internal",
            },
            clear=True,
        ):
            result = _local_eval_subprocess_env("eval-host.internal")

        for key in ("NO_PROXY", "no_proxy"):
            entries = result[key].split(",")
            self.assertIn("existing.internal", entries)
            self.assertIn("127.0.0.1", entries)
            self.assertIn("localhost", entries)
            self.assertIn("0.0.0.0", entries)
            self.assertIn("eval-host.internal", entries)
            self.assertEqual(entries.count("localhost"), 1)
        self.assertEqual(result["HTTP_PROXY"], "http://proxy.invalid:8080")


if __name__ == "__main__":
    import unittest

    unittest.main()
