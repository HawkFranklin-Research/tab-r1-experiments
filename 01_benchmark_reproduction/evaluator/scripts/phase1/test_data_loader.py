from data_loader import DataLoader
import os
from pathlib import Path

def run_test():
    loader = DataLoader(seed=42)
    
    # Path to classification dataset (Australian)
    cls_path = str(Path(__file__).resolve().parents[4] / "ev_tabpfn/src/ev_tabpfn/datasets/sample/australian_sample.csv")
    
    print("\n--- Testing Classification Load (Australian) ---")
    if os.path.exists(cls_path):
        container = loader.load_local_csv(cls_path)
        print(f"Task: {container.task_type}")
        print(f"Target: {container.target_name}")
        print(f"Features: {len(container.feature_names)}")
        print(f"Train samples: {len(container.X_train)}")
        assert container.task_type == "binary"
        assert len(container.X_train) > 0
    else:
        print(f"File not found: {cls_path}")

    # Path to a regression dataset from the dummy sets
    reg_path = str(Path(__file__).resolve().parents[4] / "ev_tabpfn/src/ev_tabpfn/datasets/sample/linear_relation_2d_sample.csv")
    
    print("\n--- Testing Regression Load (Linear Relation 2D) ---")
    if os.path.exists(reg_path):
        container = loader.load_local_csv(reg_path)
        print(f"Task: {container.task_type}")
        print(f"Target: {container.target_name}")
        print(f"Features: {len(container.feature_names)}")
        print(f"Train samples: {len(container.X_train)}")
        assert container.task_type == "regression"
        assert len(container.X_train) > 0
    else:
        print(f"File not found: {reg_path}")

    print("\n✅ Phase 1: DataLoader tests passed!")

if __name__ == "__main__":
    run_test()
