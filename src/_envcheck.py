import importlib, sys
print("python:", sys.version.split()[0], sys.executable)
for m in ["pandas","numpy","matplotlib","seaborn","sklearn","yaml","pyarrow","tqdm","requests"]:
    try:
        mod = importlib.import_module(m)
        v = getattr(mod, "__version__", "?")
        print(f"  OK    {m:12s} {v}")
    except Exception as e:
        print(f"  MISS  {m:12s} {e}")
