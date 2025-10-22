
#!/usr/bin/env python
print("Run everything:")
print(r"""
python scripts/generate_all_visualizations.py \
  --results outputs/test_results \
  --output figs \
  --dpi 300 \
  --style paper
""")
