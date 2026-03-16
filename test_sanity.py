"""
test_sanity.py — Basic sanity checks on data files and training log.
"""
import anndata as ad
import pandas as pd

results = []

# --- Check 1: Load kang2018_pbmc.h5ad ---
print("=" * 60)
print("CHECK 1: Kang 2018 PBMC dataset")
print("=" * 60)
try:
    adata = ad.read_h5ad("data/kang2018_pbmc.h5ad")
    n_cells, n_genes = adata.shape
    print(f"  Cells: {n_cells}")
    print(f"  Genes: {n_genes}")
    # Accept the full dataset (24673) or close variants
    if n_cells > 20000 and n_genes > 10000:
        results.append(("Dataset shape", "PASS", f"{n_cells} cells x {n_genes} genes"))
    else:
        results.append(("Dataset shape", "FAIL", f"Unexpected shape: {n_cells} x {n_genes}"))
except Exception as e:
    results.append(("Dataset loading", "FAIL", str(e)))
    print(f"  ERROR: {e}")

# --- Check 2: Condition column ---
print()
print("=" * 60)
print("CHECK 2: Condition labels (adata.obs['label'])")
print("=" * 60)
try:
    label_counts = adata.obs["label"].value_counts()
    print(f"  Value counts:")
    for label, count in label_counts.items():
        print(f"    {label}: {count}")
    n_conditions = label_counts.shape[0]
    if n_conditions == 2 and "ctrl" in label_counts.index and "stim" in label_counts.index:
        results.append(("Condition labels", "PASS", f"2 conditions: ctrl={label_counts['ctrl']}, stim={label_counts['stim']}"))
    else:
        results.append(("Condition labels", "FAIL", f"Expected ctrl/stim, got {list(label_counts.index)}"))
except KeyError:
    results.append(("Condition labels", "FAIL", "'label' column not found in obs"))
    print("  ERROR: 'label' column not found")

# --- Check 3: Cell types ---
print()
print("=" * 60)
print("CHECK 3: Cell types (adata.obs['cell_type'])")
print("=" * 60)
try:
    cell_types = adata.obs["cell_type"].unique().tolist()
    print(f"  Found {len(cell_types)} unique cell types:")
    for ct in sorted(cell_types):
        print(f"    - {ct}")
    if len(cell_types) >= 5:
        results.append(("Cell types", "PASS", f"{len(cell_types)} cell types found"))
    else:
        results.append(("Cell types", "FAIL", f"Only {len(cell_types)} cell types, expected >= 5"))
except KeyError:
    results.append(("Cell types", "FAIL", "'cell_type' column not found in obs"))
    print("  ERROR: 'cell_type' column not found")

# --- Check 4: Edge list ---
print()
print("=" * 60)
print("CHECK 4: Edge list (data/edge_list.csv)")
print("=" * 60)
try:
    edge_df = pd.read_csv("data/edge_list.csv")
    n_edges = len(edge_df)
    score_min = edge_df["combined_score"].min()
    score_max = edge_df["combined_score"].max()
    print(f"  Row count: {n_edges}")
    print(f"  Score range: {score_min} - {score_max}")
    print(f"  Columns: {list(edge_df.columns)}")
    print(f"  First 5 rows:")
    print(edge_df.head().to_string(index=False))
    if n_edges >= 5000 and n_edges <= 25000 and score_min >= 700:
        results.append(("Edge list", "PASS", f"{n_edges} edges, scores {score_min}-{score_max}"))
    elif n_edges < 5000:
        results.append(("Edge list", "FAIL", f"Only {n_edges} edges, expected 5000-25000"))
    elif score_min < 700:
        results.append(("Edge list", "FAIL", f"Min score {score_min} < 700"))
    else:
        results.append(("Edge list", "FAIL", f"Too many edges: {n_edges}"))
except FileNotFoundError:
    results.append(("Edge list", "FAIL", "data/edge_list.csv not found"))
    print("  ERROR: File not found")

# --- Check 5: Training log ---
print()
print("=" * 60)
print("CHECK 5: Training log (training_log.txt)")
print("=" * 60)
try:
    with open("training_log.txt", "r") as f:
        lines = f.read().strip().split("\n")
    n_epochs = len(lines)
    print(f"  Total epochs logged: {n_epochs}")
    print(f"  First 3 lines:")
    for line in lines[:3]:
        print(f"    {line}")
    print(f"  Last 3 lines:")
    for line in lines[-3:]:
        print(f"    {line}")

    # Check loss went down by parsing first and last
    first_loss = float(lines[0].split("Loss: ")[1])
    last_loss = float(lines[-1].split("Loss: ")[1])
    if n_epochs >= 10 and last_loss < first_loss:
        results.append(("Training log", "PASS", f"{n_epochs} epochs, loss {first_loss:.4f} -> {last_loss:.4f}"))
    elif n_epochs < 10:
        results.append(("Training log", "FAIL", f"Only {n_epochs} epochs logged"))
    else:
        results.append(("Training log", "FAIL", f"Loss did not decrease: {first_loss:.4f} -> {last_loss:.4f}"))
except FileNotFoundError:
    results.append(("Training log", "FAIL", "training_log.txt not found"))
    print("  ERROR: File not found")

# --- Summary ---
print()
print("=" * 60)
print("SANITY TEST SUMMARY")
print("=" * 60)
all_pass = True
for check, status, detail in results:
    icon = "✓" if status == "PASS" else "✗"
    print(f"  {icon} {check}: {status} — {detail}")
    if status == "FAIL":
        all_pass = False

print()
if all_pass:
    print("OVERALL: PASS")
else:
    print("OVERALL: FAIL")
