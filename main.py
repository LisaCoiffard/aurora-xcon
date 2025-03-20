import sys

from train import (
    aurora_trainer,
    ga_trainer,
    me_trainer,
    pga_aurora_trainer,
    pga_me_trainer,
    td3_trainer,
    jedi_trainer,
)


def main():
    if len(sys.argv) < 3:
        print("Usage: python main.py <algo> [hydra_args...]")
        sys.exit(1)

    algo = sys.argv[2] if len(sys.argv) > 2 else sys.argv[1]
    hydra_args = sys.argv[3:] if len(sys.argv) > 3 else sys.argv[2:]

    # Map the algorithm name to the corresponding module
    algo_mapping = {
        "me": me_trainer.main,
        "jedi": jedi_trainer.main,
        "aurora": aurora_trainer.main,
        "ga": ga_trainer.main,
        "pga_me": pga_me_trainer.main,
        "pga_aurora": pga_aurora_trainer.main,
        "td3": td3_trainer.main,
    }

    if algo not in algo_mapping:
        print(f"Error: Algorithm '{algo}' not recognized.")
        sys.exit(1)

    # Dynamically call the selected trainer's train function
    train_func = algo_mapping[algo]

    # Use Hydra's command-line arguments and pass them directly
    # Equivalent to passing `python me_trainer.py +arg=value`
    sys.argv = [sys.argv[0]] + hydra_args

    train_func()  # Run the selected trainer


if __name__ == "__main__":
    main()
