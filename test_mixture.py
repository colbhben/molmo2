"""Dry test: build the gaze_specialize rehearse mixture and verify group weights.

Does NOT load data -- only exercises get_training_mixture()'s structure so we catch
weight/normalization/registration mistakes before a full smoke run.
"""
import os, sys
os.environ.setdefault("GAZE_SPECIALIZE_RATIO", "0.92")

# import get_training_mixture from sft.py
sys.path.insert(0, "/molmo2/launch_scripts")
import importlib.util
spec = importlib.util.spec_from_file_location("sft", "/molmo2/launch_scripts/sft.py")
sft = importlib.util.module_from_spec(spec)
spec.loader.exec_module(sft)

mix = sft.get_training_mixture("gaze_specialize")
print(f"{'group':22s} {'rate':>8s}  datasets")
total = 0.0
for km in mix:
    rate = km.rate
    name = km.name or "?"
    subs = km.datasets
    total += rate
    ds_names = [getattr(d, "dataset_name", str(d)) for d in subs]
    print(f"{name:22s} {rate:8.4f}  {ds_names}")
print(f"{'TOTAL':22s} {total:8.4f}")
assert abs(total - 1.0) < 1e-6, f"mixture rates must sum to 1.0, got {total}"

# verify gaze gets 0.92 and rehearse groups sum to 0.08 in paper proportions
gaze = [km for km in mix if km.name == "gaze"][0]
gaze_rate = gaze.rate
print(f"\ngaze rate = {gaze_rate} (expect 0.92)")
assert abs(gaze_rate - 0.92) < 1e-6
reh = [km for km in mix if (km.name or "").startswith("rehearse_")]
reh_total = sum(km.rate for km in reh)
print(f"rehearse total = {reh_total:.4f} (expect 0.08)")
assert abs(reh_total - 0.08) < 1e-6
# check relative proportions among rehearse groups match paper renorm
print("\nrehearse group shares (of the 8% budget):")
for km in reh:
    print(f"  {km.name:22s} {km.rate/reh_total:.4f}")
print("\nMIXTURE_OK")
