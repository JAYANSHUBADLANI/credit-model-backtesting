# Swapping the fixture for a real extract

Everything in this project runs against `scripts/make_fixture.py` output. This is what changes
to point it at the real thing. None of it is a rewrite; the layout was kept in one place
precisely so this stays a configuration job.

## 1. Get the data

Either publisher works. Both are free and both require a registration that has to be completed
by a person.

- **Freddie Mac Single Family Loan-Level Dataset.** Prefer this one to start. It publishes
  **sample files**: a random 50,000 loans per origination year with matching monthly
  performance. That is the right size for this project and it is the difference between a
  weekend and a month of parsing.
- **Fannie Mae Single-Family Loan Performance**, via Data Dynamics. Equivalent, larger, and
  organised by origination quarter.

Do not pull the full population, and do not pull every year. Scope is the main way this project
fails. The vintages in `config/config.yaml` are the ones the analysis needs.

## 2. Read the current layout document first

**Do not trust the column positions in `src/loans.py`.** They describe the fixture, which was
written to a plausible layout, and both publishers have changed their real layout at least once.
Open the layout document that ships with the extract and rewrite these two dictionaries:

```python
ORIGINATION_LAYOUT = {"loan_id": 0, "orig_date": 1, "credit_score": 2, ...}
PERFORMANCE_LAYOUT = {"loan_id": 0, "period": 1, "loan_age": 2, ...}
```

They map a column name to its zero based position. Every other module reads by name, so nothing
downstream changes.

## 3. Check the sentinels

`SENTINELS` in `src/loans.py` lists the values that mean "not collected" rather than a
measurement, a credit score of 9999, a debt to income of 999. Read naively, 9999 is the best
applicant in the book, it survives quantile binning as a real bin, and the card allocates points
to a missing data code. Confirm the values this extract uses and add any that are missing.

## 4. Check the zero balance codes

`config/config.yaml` names which codes are a credit loss and which are a voluntary payoff:

```yaml
credit_loss_zero_balance_codes: ["02", "03", "09", "15"]
prepaid_zero_balance_codes: ["01"]
```

Getting these wrong moves loans between "default" and "prepaid", which changes both the target
and the competing risk. The config refuses a code listed in both.

## 5. Point the globs at the files

```yaml
data:
  origination_glob: "data/raw/historical_data_*.txt"
  performance_glob: "data/raw/historical_data_time_*.txt"
```

## 6. Set the vintages

```yaml
vintages:
  fit: [2004, 2005]
  backtest: [2004, 2005, 2006, 2007, 2008, 2015, 2016, 2017]
```

The fit vintages must appear in the backtest set; the config raises if they do not, because the
fit period has to sit on the same axis as everything else for there to be a baseline at all.

## 7. Run the acceptance test before anything else

```bash
make train
```

and read the completeness table it prints. **Before looking at a single model number, check that
the realised 24 month default rate for a benign vintage and a stress vintage are clearly
different.** If 2005 and 2007 come out similar, the parser or the target definition is wrong and
everything built on top of it would be wrong too. Nothing else is worth debugging until that
holds.

Also check the `unobservable` column. If a vintage you expected to analyse is being dropped
wholesale, the extract does not reach far enough past it to observe the full window, and it
belongs out of the backtest set rather than in it with a caveat.

## 8. Expect the intervals to close

The dominant limitation of the fixture run is sample size: 4,000 loans per vintage gives a Gini
spread of 0.17, wide enough that seven of eight vintages returned "not proven" on
discrimination. At 50,000 loans per vintage that spread roughly triples in precision, and the
degradation question becomes answerable rather than merely honest.

Re-run `make noise` on the real extract before quoting any degradation verdict. The floor is a
property of the sample, not of the code, and every verdict in this project is measured against
it.
