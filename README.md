# Credit model backtesting and outcome monitoring

Watching whether a credit scorecard is still **right**, measured against realised defaults over
time, rather than whether its inputs have moved.

This is the second half of a pair. The first,
[`credit-scorecard-service`](../credit-scorecard-service), serves a scorecard behind an API and
watches its inputs and outputs with stability indices. It ends on an admission: nothing in it
monitors whether the model is still correct, only whether the population has shifted. A card can
be perfectly stable and quietly stop working. This project is the other half, and its most
useful result turned out to be a criticism of the first one.

## Read this before any number below

**Every figure in this README was produced from a synthetic extract that I generated.** The real
Freddie Mac and Fannie Mae loan level datasets are free but sit behind a registration that an
automated build cannot complete, so rather than ship a pipeline that had never been run — a
mistake I have already made once and written up — the whole thing is exercised end to end
against `scripts/make_fixture.py`, which writes the same pipe delimited layout the real extracts
use.

So:

| | |
|---|---|
| **Real** | The methodology, the code, the target construction, the measurements, the trigger logic, the tests, and the fact that all of it runs. |
| **Synthetic** | Every loan. Every default. Every number quoted below. |

The fixture plants a specific failure: a stable ranking relationship, a baseline default rate
that jumps in the stress vintages by more than the characteristics account for, a population
that shifts underneath it, and one coefficient that genuinely moves. The backtest recovering
that is evidence that the measurement works and the triggers fire on a known signal. **It is not
evidence about mortgages, about 2007, or about anything in the world.** Point the same commands
at a real extract and they produce numbers that mean something. `docs/using_real_data.md` is what
to change.

## The finding

Three things came out of this that I did not expect when I planned it.

**1. Calibration failed catastrophically while discrimination never provably moved.**

The frozen 2004–2005 card was still ranking loans in 2007 about as well as it ever had — Gini
0.4468 against a day one 0.4890, a gap that sits comfortably inside the measured noise. Over the
same book it predicted a 3.06% default rate against a realised 13.15%. A factor of **4.3**. A
monitoring setup watching the Gini would have reported no problem at all, for years, through a
complete failure of the model's level.

**2. The relative error was worst on the applicants the card liked most.**

Calibration by score band in 2007, best band last:

| band | loans | predicted | realised | ratio |
|---|---|---|---|---|
| 0 (worst) | 761 | 9.03% | 30.88% | 3.4× |
| 3 | 409 | 1.82% | 9.54% | 5.2× |
| 6 | 319 | 0.70% | 5.33% | 7.6× |
| 9 (best) | 202 | 0.19% | 3.47% | **18.6×** |

The card was wrong everywhere and *proportionally* most wrong about the loans it was most
confident in. In absolute terms the bad bands still lost more; in the terms a pricing or
provisioning model actually uses, the prime end of the book was the part nobody could trust.

**3. The early warning signal bought nothing on this book.**

This is the part that criticises the other project. The score stability index reached only
`warn` (0.1406), and not until the 2007 vintage — one vintage *after* calibration had already
breached. The run reports it plainly:

> the outcome breached first. The stability index did not move until a later vintage, so on
> this book it bought no warning at all.

The reason is structural, not a tuning failure. A stability index compares who is applying now
with who applied before. It can only see a change in **composition**. The failure planted here
is mostly a change in the **level of risk attached to unchanged characteristics** — the same
borrower, the same LTV, a different world. That is invisible to input monitoring by
construction, however the thresholds are set.

That does not make input monitoring useless. It makes its scope precise: it catches the
population moving, and it is silent on the world moving. Those are different failures and only
one of them has an early warning.

## The system

```
     GSE loan level extract (pipe delimited, headerless, origination + monthly performance)
                                         |
                          src/loans.py   layout map, sentinels -> NaN
                                         |
                          src/target.py  24 month window, D180 or credit loss,
                                         prepayment kept separate, truncation dropped
                                         |
                    +--------------------+--------------------+
                    |                                         |
          fit vintages 2004-2005                    every backtest vintage
                    |                                         |
             [ make train ]                                   |
      frozen champion, never refitted  --------------->  [ make backtest ]
                                                              |
                    +-----------------------+-----------------+------------------+
                    |                       |                 |                  |
             discrimination           calibration        stability          challenger
             + bootstrap CI           + Wilson CI        PSI / CSI        feasible vs oracle
                    |                       |                 |                  |
                    +-----------+-----------+--------+--------+                  |
                                |                    |                           |
                          src/triggers.py      src/stability.py bridge           |
                        recalibrate vs refit   one dated timeline                |
                                |                    |                           |
                                +--------------------+---------------------------+
                                                     |
                                            reports/  +  dashboard
```

## Quick start

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements-dev.txt
make demo
```

`make demo` generates the fixture, fits the champion, measures the noise floor, and runs the
backtest. It takes well under a minute and writes everything quoted here to `reports/`.

```bash
make dashboard
```

## The three decisions that had to be got right

The model is deliberately the boring part. These are not.

### The performance window is fixed for every vintage

24 months from origination, identically, everywhere. A 2008 origination observed at 12 months
and a 2005 origination observed at 24 months are not comparable, and a chart that puts them side
by side is measuring seasoning rather than credit. It is read from config, applied in one place,
and asserted after filtering.

### Truncation drops the vintage, not the survivors

This one produced a wrong answer before I caught it, and it is the defect I would point an
interviewer at first.

The obvious definition of a usable loan is "it terminated, or it reached the end of the window".
Applied to a vintage the extract only partly covers, that keeps every loan that defaulted early
— defaulting *is* a termination — and drops every loan still quietly performing. The survivors
are discarded and the failures are kept.

On the fixture it inflated the newest vintage from **0.90% to 2.1%**, putting it above vintages
that were genuinely better and reversing the ranking of the last three years of the book.

So completeness is a property of the **vintage**, decided by its origination date alone, before
anything about an individual loan is looked at. A vintage the extract cannot cover to term is
dropped whole. Half the 2017 fixture goes this way and the completeness report says so.
`tests/test_target.py::test_truncation_drops_the_whole_vintage_not_just_the_survivors` is the
regression test.

### Prepayment is a competing risk and is never folded into the target

A loan that prepaid at month eight never had the opportunity to default. Mortgage prepayment is
not a rounding error — a quarter to a third of every vintage here leaves early. Both rates are
reported side by side and every table says which one it is:

| vintage | complete | defaults | prepaid | naive rate | survivors only |
|---|---|---|---|---|---|
| 2004 | 3,999 | 81 | 1,160 | 2.03% | 2.85% |
| 2005 | 4,000 | 88 | 1,083 | 2.20% | 3.02% |
| 2006 | 4,000 | 214 | 737 | 5.35% | 6.56% |
| 2007 | 4,000 | 526 | 528 | 13.15% | 15.15% |
| 2008 | 4,000 | 282 | 696 | 7.05% | 8.54% |
| 2015 | 4,000 | 44 | 1,098 | 1.10% | 1.52% |
| 2016 | 4,000 | 43 | 1,229 | 1.08% | 1.55% |
| 2017 | 1,994 | 18 | 514 | 0.90% | 1.22% |

Proper survival treatment is a stretch goal. Naming the bias and its direction is not.

## The champion

The weight of evidence and logistic methodology is lifted unchanged from the scorecard project
this accompanies, along with `binning.py` and `scorecard.py` themselves. Reusing it is the
point: a challenger fitted by a different procedure is not a challenger, it is a confound.

Fitted on 2004 and 2005, then **frozen and never refitted**. 8 of 15 candidate characteristics
retained: `credit_score`, `state`, `orig_cltv`, `dti`, `orig_rate`, `mi_pct`, `property_type`,
`purpose`. Day one holdout, 2,393 loans at a 2.26% bad rate: **Gini 0.4890, KS 0.3987, AUC
0.7445**.

The fit period is benign by choice, which is hindsight, and the README says so rather than
waiting to be asked. What it buys is a clean baseline.

One derived characteristic was added: `second_lien_gap`, combined LTV minus LTV, the silent
second lien behind the first. A borrower at 80 LTV with a 20 point gap put nothing down; one at
80 LTV with no gap put twenty percent down. Same first lien, different credit.

## The noise floor, measured before anything is claimed

`make noise` resamples one unchanged population and asks how far the Gini wanders. The answer
decides how much of this project is allowed to be stated as fact.

| vintage size | mean defaults | p05 | p50 | p95 | spread |
|---|---|---|---|---|---|
| 500 | 11 | 0.2287 | 0.5018 | 0.7237 | 0.4950 |
| 1,000 | 23 | 0.2858 | 0.5026 | 0.6864 | 0.4006 |
| 2,000 | 45 | 0.3514 | 0.4934 | 0.6119 | 0.2606 |
| **4,000** | **92** | **0.4081** | **0.4896** | **0.5774** | **0.1693** |
| 8,000 | 180 | 0.4310 | 0.4856 | 0.5498 | 0.1188 |

At a 4,000 loan vintage with a 2% default rate, the Gini's 90% spread is **0.17**. Discrimination
is estimated from the defaults, not the loans, so that vintage is really a sample of ninety
events, and ninety events do not pin a Gini to two decimal places.

This is why **every degradation verdict here is a non-overlap of intervals, never a comparison
of point estimates**, and why the honest verdict on most vintages is "not proven":

| vintage | n | bad rate | Gini | interval | vs day one | verdict |
|---|---|---|---|---|---|---|
| 2004 | 1,211 | 1.82% | 0.4934 | [0.318, 0.665] | +0.004 | no degradation |
| 2005 | 1,182 | 2.71% | 0.4824 | [0.296, 0.643] | −0.007 | not proven |
| 2006 | 4,000 | 5.35% | 0.4243 | [0.367, 0.485] | −0.065 | not proven |
| 2007 | 4,000 | 13.15% | 0.4468 | [0.408, 0.488] | −0.042 | not proven |
| 2008 | 4,000 | 7.05% | 0.4161 | [0.366, 0.472] | −0.073 | not proven |
| 2015 | 4,000 | 1.10% | 0.3980 | [0.249, 0.552] | −0.091 | not proven |
| 2016 | 4,000 | 1.08% | 0.5910 | [0.488, 0.693] | +0.102 | no degradation |
| 2017 | 1,994 | 0.90% | 0.4318 | [0.218, 0.633] | −0.057 | not proven |

The fit vintages appear as their **holdout rows only**. Scoring the loans the card was fitted on
and putting that on the same axis would inflate the baseline every later vintage is judged
against.

A reader wanting a story would take "Gini fell from 0.49 to 0.42" from that table. It is not
there. Nothing in this project ever proved discrimination degraded, and the plainest reading is
that at these sample sizes it could not have.

## Calibration

| vintage | n | defaults | predicted | realised | 95% interval | ratio | ratio low |
|---|---|---|---|---|---|---|---|
| 2004 | 1,211 | 22 | 2.01% | 1.82% | [1.20, 2.74] | 0.91 | 0.60 |
| 2005 | 1,182 | 32 | 2.07% | 2.71% | [1.92, 3.80] | 1.31 | 0.93 |
| 2006 | 4,000 | 214 | 2.73% | 5.35% | [4.69, 6.09] | 1.96 | 1.72 |
| 2007 | 4,000 | 526 | 3.06% | 13.15% | [12.14, 14.23] | **4.30** | 3.97 |
| 2008 | 4,000 | 282 | 2.09% | 7.05% | [6.30, 7.89] | 3.38 | 3.02 |
| 2015 | 4,000 | 44 | 2.26% | 1.10% | [0.82, 1.47] | 0.49 | 0.36 |
| 2016 | 4,000 | 43 | 2.07% | 1.08% | [0.80, 1.45] | 0.52 | 0.39 |
| 2017 | 1,994 | 18 | 2.21% | 0.90% | [0.57, 1.42] | 0.41 | 0.26 |

A breach has to clear its own interval, so 2005's point ratio of 1.31 against a 1.25 tolerance is
not a breach: the interval reaches below it.

**The test is two sided.** Under prediction is the dangerous direction — losses arriving that
were not provisioned for. Over prediction is the expensive one: 2015 through 2017 predict roughly
double the risk that materialises, which means declining profitable business every day the card
is left alone. Calling that "conservative" rather than "wrong" is how it survives for years.

## The bridge

Every signal on one timeline, each dated to when it could first have been known.

| vintage | PSI | status | knowable | Gini | proven loss | ratio | breach | direction | outcome knowable |
|---|---|---|---|---|---|---|---|---|---|
| 2004 | 0.0057 | ok | 2004-12 | 0.4934 | no | 0.91 | no | | 2006-12 |
| 2005 | 0.0049 | ok | 2005-12 | 0.4824 | no | 1.31 | no | | 2007-12 |
| 2006 | 0.0708 | ok | 2006-12 | 0.4243 | no | 1.96 | **yes** | under | 2008-12 |
| 2007 | 0.1406 | **warn** | 2007-12 | 0.4468 | no | 4.30 | **yes** | under | 2009-12 |
| 2008 | 0.0049 | ok | 2008-12 | 0.4161 | no | 3.38 | **yes** | under | 2010-12 |
| 2015 | 0.0943 | ok | 2015-12 | 0.3980 | no | 0.49 | **yes** | over | 2017-12 |
| 2016 | 0.0430 | ok | 2016-12 | 0.5910 | no | 0.52 | **yes** | over | 2018-12 |
| 2017 | 0.0649 | ok | 2017-12 | 0.4318 | no | 0.41 | **yes** | over | 2019-12 |

The arithmetic that makes this worth building: a stability index on the 2007 book is computable
during 2007, because it needs nothing but the applications. Its 24 month outcome is not knowable
until the end of 2009. Where input monitoring fires, it fires two years early. Here it did not
fire in time, and the timeline is the evidence rather than an assertion either way.

`orig_rate` also produced the largest characteristic index in the project, 8.60 in 2017, because
rates fell between the eras. It coincided with the best-calibrated stretch of the whole backtest.
A large CSI is not a prediction of anything.

## Triggers: recalibrate and refit are different actions

The other project asked to be judged on what its alerting declines to send. Same argument, one
layer up, resting on one distinction:

- **Level wrong, ranking intact → recalibrate.** An intercept adjustment. Nothing needs
  revalidating, the points table a committee approved still stands, the cutoffs keep their
  meaning.
- **Ranking broken → refit.** Expensive. Invalidates the cutoffs, the policy rules on top of
  them, and the documentation, and needs the whole approval cycle again.

Reporting "the model is broken" when the honest finding is "the level needs moving" is the same
failure as sending sixteen alerts for one drift event, and it costs more: not a page at 3am, but
a quarter of model risk work and a policy freeze.

The rules: nothing fires on a single vintage; a calibration breach must clear its interval;
degradation must clear the measured noise floor; **when both fire, the recommendation is refit
alone, never both**, because a refit recalibrates by construction; and a cooldown suppresses a
repeat until there has been a chance to act.

Eight vintages produced **two** recommendations, both recalibrate:

| vintage | breach | run | action | note |
|---|---|---|---|---|
| 2006 | under predicting | 1 | — | below the persistence requirement |
| 2007 | under predicting | 2 | **recalibrate** | level is wrong, not the order |
| 2008 | under predicting | 3 | — | cooldown active until 2009 |
| 2015 | over predicting | 4 | **recalibrate** | level is wrong, not the order |
| 2016–2017 | over predicting | 5–6 | — | cooldown active until 2017 |

Every suppressed breach is still written to the audit trail with the reason. A monitoring system
that silently drops signals is indistinguishable from one that never saw them.

### What recalibration does, and what it cannot do

Applied to 2007, the worst vintage:

```
realised 0.1315 against predicted 0.0306      ratio 4.2985
intercept offset +1.711 moves predicted to 0.1315   ratio 1.0000
Gini before 0.446817   after 0.446817   change 0.000000000
```

The Gini is unchanged to nine decimal places, because an intercept shift is monotone in the log
odds and no pair of loans can swap order. That is not a curiosity. It is the entire justification
for treating recalibration as the cheap action: it cannot change any ordering, so it cannot
invalidate anything that was approved on the basis of one.

## Would retraining have helped, and could it have been done

A challenger sitting in front of the 2007 book can only have been fitted on vintages whose
outcomes were already known at the start of 2007. With a 24 month window that is 2004, and
nothing later. Fitting it on 2006 — which is what "retrain on recent data" means in practice —
hands the backtest two years of information the decision maker did not have.

So two challengers are fitted at every vintage: **feasible**, respecting that constraint, and
**oracle**, ignoring it and labelled infeasible everywhere it appears.

| vintage | challenger | trained on | Gini Δ | ratio: champion → challenger | reading |
|---|---|---|---|---|---|
| 2007 | feasible | [2004] | −0.044 | 4.30 → 4.42 | ranking worse, level no better |
| 2007 | oracle | [2005, 2006] | +0.082 | 4.30 → 2.65 | ranking improved, level closer but still off |
| 2008 | feasible | [2004, 2005] | +0.016 | 3.38 → 3.27 | essentially unchanged, level closer but still off |
| 2008 | oracle | [2006, 2007] | +0.086 | 3.38 → 1.09 | ranking improved, level closer to right |
| 2015 | feasible | [2007, 2008] | +0.108 | 0.49 → 0.21 | ranking improved, level no better |

Three things fall out.

**Retraining with the information actually available would have made 2007 slightly worse**, on
both axes. The feasible challenger had one benign vintage to learn from and learned the same
thing the champion already knew.

**Even the oracle, with two years of forbidden information, was still 2.6× off on 2007.** No
amount of retraining fixes a level failure while it is happening. Recalibration does, and only
after the outcome is known.

**Retraining on the crisis leaves you badly over conservative afterwards.** The feasible
challenger for 2015, forced onto 2007–2008, predicts nearly five times the risk that
materialises. Its ranking is better than the champion's and its level is far worse.

### Swap set

If the 2007 recommendation had been acted on, **1,200 of 4,000 loans change band, 30%**. Both
directions matter: 28 loans move from approve to decline and defaulted at 28.6%, which is the
benefit; 62 move from decline to approve and defaulted at 14.5%, which is the risk taken on. A
swap that moves a third of the book had better be worth the approval cycle.

## Tests

60 tests, `make test`, in about three seconds.

| file | what it holds |
|---|---|
| `test_target.py` | window enforcement, the truncation regression, prepay vs default precedence, silent credit losses, unreported status |
| `test_metrics.py` | AUC tie handling, Wilson on small samples, bootstrap reproducibility, refusing unsupported confidence levels |
| `test_triggers.py` | persistence, cooldown, folding, recalibration's Gini invariance, unreachable levels |
| `test_stability.py` | PSI edge cases, the timing arithmetic, two sided breaches, and the "outcome breached first" reading |
| `test_challenger.py` | the information lag, pinned so a future edit cannot quietly hand the backtest hindsight |
| `test_loans_and_config.py` | sentinels, duplicate ids, config validation |
| `test_backtest.py` | verdict logic, cumulative incidence vs hazard, and one end to end run on a generated extract |
| `test_dashboard.py` | the page actually executes, using Streamlit's own harness rather than an HTTP 200 |

The end to end test generates an extract, fits a card, backtests it and evaluates the triggers,
with no mocks. Nothing here is code that has only ever been read.

## Repository layout

```
src/
  config.py      typed config, and validation that refuses a meaningless backtest
  loans.py       GSE layout map, sentinel handling, period arithmetic
  target.py      the window, the truncation rule, the competing risk
  features.py    origination fields to model features, one shared path
  binning.py     monotonic WOE binning            [reused unchanged]
  scorecard.py   logistic on WOE, points scaling  [reused, two artifact fields added]
  train.py       fit the champion, freeze it; `fit_card` is shared with the challenger
  metrics.py     discrimination and calibration, both with intervals
  backtest.py    per vintage measurement, vintage curves, the degradation verdict
  stability.py   PSI and CSI against the frozen reference, and the dated bridge
  triggers.py    recalibrate vs refit, persistence, cooldown, swap sets
  challenger.py  feasible and oracle challengers
scripts/
  make_fixture.py    synthetic extract in the GSE layout, with the planted signal documented
  gini_noise.py      the measured noise floor
  run_backtest.py    the documented entrypoint behind every number here
dashboard/app.py
```

## Honest assessment

What this demonstrates: an outcome backtest that separates two failure modes that are usually
reported as one, verdicts that are interval based because the intervals were measured first, a
trigger design whose interesting decisions are about restraint and about which of two actions is
being recommended, a challenger comparison that respects what was knowable at the time, and a
selection bias I found, fixed, wrote a regression test for, and did not quietly delete.

Where it is weak:

- **The data is synthetic and so is every number.** This is the dominant limitation and no
  amount of careful methodology below it changes that. The pipeline is real and has run; the
  findings are about the pipeline.
- **The planted signal is one I designed.** That the backtest recovers it is close to circular.
  It rules out the pipeline being broken. It establishes nothing else.
- **The samples are too small for the discrimination question.** A 0.17 Gini spread at vintage
  size means this project can barely answer whether ranking degraded. Real extracts have
  hundreds of thousands of loans per vintage and the interval closes. On this fixture the honest
  verdict was "not proven" seven times out of eight.
- **Prepayment is handled by reporting two rates, not by a survival model.** Named, not solved.
- **The fit period was chosen with hindsight**, and the crisis is one everybody knows happened.
  The mechanism is what is demonstrated, never foresight.
- **The window is 24 months because I chose 24 months.** It is not calibrated against when this
  portfolio's losses actually emerge, because there is no portfolio.
- **Mortgage, and only mortgage.** The methodology carries to consumer credit; the seasoning
  profile, the risk drivers and the competing risk structure do not.
- **Nothing here monitors fairness.** Same gap as the other project, same reason: disparate
  impact needs outcome data by protected class, and a calibration ratio says the level moved,
  never that a decision was unfair.
- **No CI, no scheduling, no deployment.** Deliberate. The other project demonstrates serving;
  this one's deliverable is evidence, and rebuilding a container around it would have cost a
  week and shown nothing new.

## License

MIT, see `LICENSE`.
