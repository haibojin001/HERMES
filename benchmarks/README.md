# Pinned benchmark sources

`lock.json` fixes the exact public commits inspected for the three benchmarks and DevOps-Gym's legacy Terminal-Bench runner. Download them into ignored `hermes_data/benchmarks`:

```bash
python benchmarks/fetch.py swe_refactor terminal_bench devops_gym terminal_bench_legacy
```

SWE Refactor Bench and Terminal-Bench 4.0 use Harbor. DevOps-Gym uses the older `tb` package, whose custom-agent API differs from Harbor's. Install the runner from the pinned checkout with `pip install -e hermes_data/benchmarks/terminal_bench_legacy` and run `python benchmarks/fetch.py devops_gym --with-assets` if its Git LFS task assets are needed.

Task repositories, image layers and hidden verifier data remain in official benchmark checkouts and task containers. They are not copied into the anonymous code bundle.
