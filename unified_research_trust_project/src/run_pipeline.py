"""一键离线流水线：生成半仿真数据 → 训练模型 → 导出解释。

完整复现（含 OpenAlex）请按 README 逐步执行 download_public_reference.py。
"""

from config import DATA_DIR
from generate_data import generate_applications
from train_models import main as train_main
from explain_and_export import main as explain_main


if __name__ == "__main__":
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    generate_applications(1500, 42).to_csv(
        DATA_DIR / "applications.csv", index=False, encoding="utf-8-sig"
    )
    train_main()
    explain_main()
