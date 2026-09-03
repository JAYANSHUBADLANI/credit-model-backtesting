PYTHON ?= python3

.PHONY: help fixture train backtest noise dashboard test demo clean

help:
	@echo "make fixture    generate a synthetic extract in the GSE file layout"
	@echo "make train      fit the champion on the fit vintages and freeze it"
	@echo "make backtest   run the frozen champion across every vintage"
	@echo "make noise      measure how far the Gini moves when nothing has changed"
	@echo "make test       run the test suite"
	@echo "make dashboard  serve the Streamlit view over the reports"
	@echo "make demo       fixture, train, noise, backtest. Every number in the README."

fixture:
	$(PYTHON) scripts/make_fixture.py

train:
	$(PYTHON) -m src.train

backtest:
	$(PYTHON) scripts/run_backtest.py

noise:
	$(PYTHON) scripts/gini_noise.py

test:
	$(PYTHON) -m pytest -q

dashboard:
	$(PYTHON) -m streamlit run dashboard/app.py

# The documented entrypoint. Everything quoted in the README comes from this target.
demo: fixture train noise backtest
	@echo ""
	@echo "reports/ now holds the evidence behind every number in the README"

clean:
	rm -rf models/*.joblib models/*.json reports/*.csv reports/*.json data/raw/*.txt
