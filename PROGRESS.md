# Progress

Running status for this project. Updated as phases complete.

## Status: the pipeline is complete and runs end to end. The data is synthetic.

Every phase below is built, tested and executed. `make demo` runs the whole thing in under a
minute and writes the evidence behind every number in the README to `reports/`.

The one thing that is not done, and cannot be done from here, is the real extract. Both the
Freddie Mac and Fannie Mae loan level datasets require a registration a person has to complete.
Until that happens, every finding is a finding about the pipeline.

## Phase by phase

### Phase 1, data layer and target construction: done

- Pipe delimited GSE layout reader with the column positions in one dictionary, so pointing at a
  real extract is a layout edit rather than a rewrite. See `docs/using_real_data.md`.
- Sentinel handling: 9999 credit scores and 999 DTIs become missing rather than the top of the
  range.
- Target: 24 month fixed window, default as 180+ days delinquent or a credit loss termination,
  prepayment kept as its own column, right truncation dropped.
- **A selection bias was found and fixed here.** Defining a usable loan as "terminated or
  reached the window" keeps early defaulters and drops still-performing loans on any vintage the
  extract covers only partly. It inflated the newest fixture vintage from 0.90% to 2.1% and
  reversed the credit quality ranking of the last three years. Completeness is now a property of
  the vintage, decided by origination date before any loan level fact is consulted.

### Phase 2, the champion: done

- Weight of evidence and logistic methodology reused unchanged from
  `credit-scorecard-service`, including `binning.py` and `scorecard.py` themselves.
- Fitted on 2004-2005, frozen, never refitted. 8 of 15 characteristics retained. Day one holdout
  Gini 0.4890, KS 0.3987, AUC 0.7445 on 2,393 loans.
- `fit_card` is shared with the challenger, so a challenger cannot differ from the champion by
  procedure.
- Added one derived characteristic, `second_lien_gap`, combined LTV minus LTV.

### Phase 3, the backtesting engine: done

- Discrimination per vintage with bootstrap intervals; calibration in the large and by fixed
  score band with Wilson intervals; vintage curves as cumulative incidence.
- `scripts/gini_noise.py` measures how far the Gini moves on an unmoved population. 0.17 of
  spread at a 4,000 loan vintage. Every degradation verdict is a non-overlap of intervals
  because of that measurement.
- Fit vintages are represented by their holdout rows only, so the baseline is not inflated.

### Phase 4, stability and the bridge: done

- PSI and CSI against the frozen fit period reference, run on origination vintages.
- `signal_timing` dates every signal to when it could first have been known, which is what makes
  the lead time a number rather than an adjective.

### Phase 5, triggers and the challenger: done

- Recalibrate and refit separated, with persistence, cooldown, folding, and an audit trail that
  records every suppressed breach and why.
- Calibration breaches are two sided. Over prediction is a real failure, not conservatism.
- Feasible and oracle challengers, with the information lag enforced in code and pinned by
  `tests/test_challenger.py` so a later edit cannot quietly reintroduce hindsight.

### Phase 6, surface: done

- Streamlit dashboard reading the reports rather than recomputing, so it and the README cannot
  disagree.
- 60 tests including an end to end run that generates an extract, fits a card, backtests it and
  evaluates the triggers with no mocks.
- Single entrypoint, `make demo`.

## Things worth knowing that came out of building it

- **The early warning signal bought nothing on this book.** The score PSI reached only `warn`,
  and one vintage *after* calibration had already breached. This is structural rather than a
  tuning failure: a stability index sees composition change, and the planted failure is mostly a
  change in the risk attached to unchanged characteristics. It is the most useful result here
  and it is a criticism of the project this one accompanies.
- **Calibration failed by 4.3× while discrimination never provably moved.** The two failure
  modes are genuinely independent and a Gini-only monitor sees nothing.
- **The relative error was worst on the best applicants**, 18.6× in the top band against 3.4× in
  the bottom. Absolute losses concentrate at the bad end; proportional wrongness at the good end.
- **Recalibration leaves the Gini unchanged to nine decimal places.** That is the justification
  for treating it as the cheap action, not a curiosity.
- **Retraining with information actually available would have made 2007 worse.** The feasible
  challenger had one benign vintage to learn from. Even the oracle, with two years it should not
  have had, was still 2.6× off.
- **Retraining on a crisis leaves you over conservative for years afterwards.** The feasible 2015
  challenger, forced onto 2007-2008, predicts nearly five times the risk that materialises.
- **The noise floor decides how much can be said.** Seven of eight vintages returned "not proven"
  on discrimination. Wanting a cleaner story does not make the intervals narrower.

## Still open

1. **The real extract.** Everything else is downstream of this. `docs/using_real_data.md` is the
   procedure, and step 7 is the acceptance test to run before looking at any model number.
2. **Sample size.** At 4,000 loans per vintage the discrimination question is barely answerable.
   The Freddie Mac sample files at 50,000 per vintage are what makes it answerable.
3. **Prepayment as a competing risk is reported, not modelled.** Two rates side by side with the
   bias direction stated. A proper survival treatment is the honest next step.
4. **No fairness measurement**, for the same reason as the other project: it needs outcome data
   by protected class.
5. **Recent vintages cannot be evaluated on equal footing.** Half of the 2017 fixture is dropped
   as unobservable. That is correct behaviour, and it means the newest book is always the one
   you know least about, which is a real operational problem this project does not solve.
6. **No CI and no deployment, deliberately.** The deliverable here is evidence, not uptime.

## Known limitations, stated up front

These are in the README too, because an interviewer will find them:

- The data is synthetic. Every number is a number about the pipeline.
- The planted signal is one I designed, so recovering it is close to circular. It rules out the
  pipeline being broken and establishes nothing else.
- The fit period was chosen with hindsight and the crisis is one everybody knows happened.
- The 24 month window is a choice, not a calibration against when losses actually emerge.
- Mortgage only. The methodology carries to consumer credit; the seasoning profile, the risk
  drivers and the competing risk structure do not.
