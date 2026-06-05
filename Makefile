# ============================================================
# Makefile — scene3d project commands
# ============================================================
# Usage:
#   make lint      — format + type-check all Python files
#   make test      — run all pytest tests
#   make demo      — launch viser viewer on example scene
#   make eval      — run evaluation metrics
#   make preprocess VIDEO=path/to/video.mp4 SCENE=scene1 — extract frames
#   make clean     — remove generated outputs
# ============================================================

.PHONY: lint test demo eval preprocess clean help

# Default: show help
help:
	@echo ""
	@echo "  scene3d — 3D Scene Reconstruction from Monocular Video"
	@echo "  ======================================================="
	@echo ""
	@echo "  make preprocess VIDEO=video.mp4   Extract + filter frames"
	@echo "  make demo SCENE=scene1            Launch interactive viewer"
	@echo "  make eval SCENE=scene1            Run evaluation metrics"
	@echo "  make lint                         Format + type-check code"
	@echo "  make test                         Run all tests"
	@echo "  make clean                        Remove generated outputs"
	@echo ""

# --- Code Quality ---
lint:
	black preprocess/ geometry/ semantics/ viewer/ eval/ tests/
	isort preprocess/ geometry/ semantics/ viewer/ eval/ tests/
	mypy preprocess/ geometry/ semantics/ viewer/ eval/ --ignore-missing-imports

# --- Tests ---
test:
	pytest tests/ -v --tb=short

# --- Preprocessing ---
preprocess:
	@if [ -z "$(VIDEO)" ]; then echo "Usage: make preprocess VIDEO=path/to/video.mp4 SCENE=scene1"; exit 1; fi
	@if [ -z "$(SCENE)" ]; then echo "Usage: make preprocess VIDEO=path/to/video.mp4 SCENE=scene1"; exit 1; fi
	python -m preprocess.extract_frames --video $(VIDEO) --output_dir data/$(SCENE)/frames

# --- Interactive Viewer ---
demo:
	@if [ -z "$(SCENE)" ]; then echo "Usage: make demo SCENE=scene1"; exit 1; fi
	python -m viewer.app --scene $(SCENE)

# --- Evaluation ---
eval:
	@if [ -z "$(SCENE)" ]; then echo "Usage: make eval SCENE=scene1"; exit 1; fi
	python -m eval.metrics --scene $(SCENE)

# --- Cleanup ---
clean:
	rm -rf data/*/frames/
	rm -rf data/*/colmap/
	rm -rf data/*/masks/
	rm -rf outputs/
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete 2>/dev/null || true
