
#!/usr/bin/env python
print("Run everything:")
print(r"""
python scripts/generate_all_visualizations.py \
  --results outputs/test_results \
  --output figs \
  --dpi 300 \
  --style paper
""")

print("For an SSTrans/NTIRE output folder (targets are optional):")
print(r"""
python scripts/generate_all_visualizations.py \
  --results outputs/source_validation \
  --targets /path/to/ARAD_1K/Train_spectral \
  --output figs/sstrans \
  --dpi 300
""")
