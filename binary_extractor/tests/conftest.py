import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).parent.parent))
import pytest, yaml, shutil
from extractor.pipeline import run

@pytest.fixture(scope="session", autouse=True)
def ensure_real_templates():
    """
    One‑time per test session:
    1. Run a tiny crop through the pipeline with template_match off.
    2. Harvest high‑confidence templates via the existing harvest step.
    3. Copy template PNGs into tests/data/ for downstream tests.
    """
    import tempfile
    tests_dir = pathlib.Path(__file__).parent
    data_dir  = tests_dir / "data"
    work      = tests_dir / "harvest_tmp"
    if work.exists():
        shutil.rmtree(work)
    work.mkdir(exist_ok=True)

    # Use the poster top row crop (should already be in tests/data/)
    sample_img = data_dir / "top_row.png"
    assert sample_img.exists(), "Need a crop with clear digits for harvesting"

    # Load test cfg and disable template matching for harvest
    cfg = yaml.safe_load(open(tests_dir / "cfg_test.yaml"))
    cfg["template_match"] = False
    cfg["super_resolution"] = False

    # Run mini‑pipeline
    run(sample_img, work, cfg)

    # Copy harvested templates into tests/data/
    for tpl in work.glob("template_*_*.png"):
        shutil.copy(tpl, data_dir / tpl.name)

    # Verify both digit classes present
    assert list(data_dir.glob("template_0_*.png"))
    assert list(data_dir.glob("template_1_*.png"))

    # Clean up temp dir
    shutil.rmtree(work) 