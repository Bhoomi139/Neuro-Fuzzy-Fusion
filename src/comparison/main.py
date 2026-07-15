import warnings
from pathlib import Path
from .experiment import run_experiment

warnings.filterwarnings("ignore")


def main():
    base_dir = Path(__file__).resolve().parent.parent.parent
    
    data_dir = base_dir / "logits"
    output_dir = base_dir / "output"
    
    run_experiment(str(data_dir), str(output_dir))


if __name__ == "__main__":
    main()
