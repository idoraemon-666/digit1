import json
from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]
CONFIG_ROOT = ROOT / "configurations"


class Protocol3ConfigurationTests(unittest.TestCase):
    def test_gate2_matrix_configs_are_isolated_and_exact(self):
        outputs = set()
        for scale in ("2p50", "2p25"):
            for reference in (50, 100):
                name = (
                    "digit_writing_original_protocol3_gate2_"
                    f"scale{scale}_ref{reference}.json"
                )
                with (CONFIG_ROOT / name).open("r", encoding="utf-8") as handle:
                    config = json.load(handle)
                self.assertEqual(
                    config["protocol"],
                    "digit_writing_original_protocol3",
                )
                self.assertEqual(config["run_kind"], "protocol3_gate2")
                self.assertEqual(config["timing_mode"], "fixed_segment_timing")
                self.assertEqual(config["selected_reference_steps"], reference)
                self.assertEqual(
                    config["scale_multiplier"],
                    2.5 if scale == "2p50" else 2.25,
                )
                self.assertEqual(
                    [case["digit"] for case in config["cases"]],
                    [1, 5, 8],
                )
                self.assertEqual(
                    [case["delay_steps"] for case in config["cases"]],
                    [50, 50, 50],
                )
                self.assertEqual(
                    [case["direction_index"] for case in config["cases"]],
                    [0, 0, 0],
                )
                output = config["output"]["directory"]
                self.assertIn("digit_writing_original_protocol3", output)
                self.assertNotIn("protocol2", output)
                outputs.add(output)
        self.assertEqual(len(outputs), 4)

    def test_server_gate_sequence_enforces_the_only_medium_fallback(self):
        gate2 = (
            ROOT / "server" / "run_digit_writing_original_protocol3_gate2.sh"
        ).read_text(encoding="utf-8")
        self.assertIn('summary["classification"] == "behavior_failure"', gate2)
        self.assertIn('summary["medium_fallback_allowed"] is True', gate2)
        self.assertIn('[[ "$REFERENCE" == "50" ]] && exit 20', gate2)
        self.assertIn('exit 21', gate2)

    def test_protocol_loader_names_protocol3_explicitly(self):
        source = (ROOT / "config.py").read_text(encoding="utf-8")
        self.assertIn('"digit_writing_original_protocol3"', source)
        self.assertIn('run_kind == "protocol3_gate2"', source)

    def test_protocol2_configs_remain_protocol2_only(self):
        for path in CONFIG_ROOT.glob("digit_writing_original_protocol2_*.json"):
            with path.open("r", encoding="utf-8") as handle:
                config = json.load(handle)
            if "protocol" in config:
                self.assertEqual(
                    config["protocol"],
                    "digit_writing_original_protocol2",
                )
            output = config.get("output", {}).get("directory")
            if output is not None:
                self.assertNotIn("protocol3", output)


if __name__ == "__main__":
    unittest.main()
