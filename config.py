import configargparse
import argparse
import json

def config_parser():
    parser = configargparse.ArgumentParser()
    
    parser.add_argument("--config", is_config_file=True, help="config file path")
    parser.add_argument("--model_name", type=str, default="rnn256_softplus")
    parser.add_argument("--experiment", type=str, default="train_2link_multi")
    parser.add_argument("--protocol_config", type=str)

    return parser


def load_protocol_config(path):
    with open(path, "r", encoding="utf-8") as handle:
        config = json.load(handle)
    if config.get("protocol") not in {
        "digit_writing_original_protocol2",
        "digit_writing_original_protocol3",
    }:
        raise ValueError("not a supported digit-writing protocol configuration")
    return config


def run_protocol_config(path):
    config = load_protocol_config(path)
    run_kind = config["run_kind"]
    if run_kind == "base_training":
        from train import train_digit_base_model

        return train_digit_base_model(config)
    if run_kind == "protocol3_gate2":
        from train import train_digit_protocol3_gate2

        return train_digit_protocol3_gate2(config)
    if run_kind == "composition":
        from digit_writing.experiments import run_composition_config

        return run_composition_config(config)
    if run_kind == "transfer5":
        from digit_writing.experiments import run_transfer_config

        return run_transfer_config(config)
    raise ValueError(f"unsupported run_kind: {run_kind}")


if __name__ == "__main__":
    arguments = config_parser().parse_args()
    if arguments.protocol_config is None:
        raise ValueError("--protocol_config is required for a protocol run")
    result = run_protocol_config(arguments.protocol_config)
    if isinstance(result, dict) and result.get("run_kind") == "protocol3_gate2":
        print(f"GATE2_CLASSIFICATION={result['classification']}")
        print(f"GATE2_AUTOMATIC_METRICS_PASSED={int(result['automatic_metrics_passed'])}")
        print(f"GATE2_MEDIUM_FALLBACK_ALLOWED={int(result['medium_fallback_allowed'])}")
