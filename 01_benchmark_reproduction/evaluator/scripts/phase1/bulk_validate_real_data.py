import sys
from pathlib import Path
import os
import pandas as pd

# Add the script directory to the path
sys.path.append(str(Path(__file__).resolve().parent))
from data_loader import DataLoader

def test_real_data():
    loader = DataLoader(seed=42)
    base_dir = str(Path(__file__).resolve().parents[3] / "datasets")
    
    datasets = [
        ("ada_dataset.csv", "binary"),
        ("australian_dataset.csv", "binary"),
        ("blood_transfusion-service-center.csv", "binary"),
        ("car.csv", "multiclass"),
        ("chum.csv", "binary"),
        ("cmc.csv", "multiclass"),
        ("credit-g.csv", "binary")
    ]
    
    print("\n--- Bulk Validation on Real Classification Data ---")
    results = []
    for filename, expected_task in datasets:
        path = os.path.join(base_dir, filename)
        try:
            container = loader.load_local_csv(path)
            results.append({
                "Dataset": filename,
                "Task": container.task_type,
                "Expected": expected_task,
                "Samples": container.metadata["total_samples"],
                "Features": container.metadata["n_features"],
                "Success": "✅" if container.task_type == expected_task else "❌"
            })
        except Exception as e:
            results.append({
                "Dataset": filename,
                "Task": "FAILED",
                "Expected": expected_task,
                "Success": f"❌ {e}"
            })

    # Print Summary Table
    df_res = pd.DataFrame(results)
    print(df_res.to_string(index=False))
    
    # Check if any failed
    assert all(r["Success"] == "✅" for r in results), "Some real datasets failed validation!"
    print("\n✅ All real-world classification datasets passed Phase 1 validation!")

if __name__ == "__main__":
    test_real_data()
