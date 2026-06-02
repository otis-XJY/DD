from config import DATA_DIR
from generate_data import generate_applications
from train_models import main as train_main
from explain_and_export import main as explain_main


if __name__ == "__main__":
    # Keep this entrypoint minimal. Detailed parameters are available in each script.
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    generate_applications(1500, 42).to_csv(DATA_DIR / "applications.csv", index=False, encoding="utf-8-sig")
    train_main()
    explain_main()
