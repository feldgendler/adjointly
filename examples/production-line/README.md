# Example: maintenance on a production line

A worked, end-to-end use of adjointly on a small but realistic optimisation problem. Run it with

```sh
npm run example
```

## The problem

Three stations work in series over a 480-minute shift, with buffers between them. Raw material
arrives at a varying rate (deliveries run short twice), and parts finished while the packing line
downstream is down fetch less. Every station wears while it runs, and its yield of good parts falls
as it wears; a minute of maintenance pauses the station and removes some wear. Profit is the value
of the good parts sold, minus a cost per maintenance minute and per scrapped part.

The decisions are discrete: six maintenance windows (two per station), each a start minute and a
length, with 120 maintenance minutes in the shift in all. A move either shifts one window by a few
minutes or moves a few minutes from one window to another.

| File                                 | What is in it                                                                    |
| ------------------------------------ | -------------------------------------------------------------------------------- |
| [model.ts](model.ts)                 | the minute-by-minute simulation, written once against `Arithmetic`               |
| [schedule.ts](schedule.ts)           | windows, pause flags, the candidate moves and their first-order estimates        |
| [search.ts](search.ts)               | plain evaluation, the gradient, forward-mode sensitivities, and the local search |
| [main.ts](main.ts), [run.ts](run.ts) | the walkthrough below, and the script that runs it                               |

## What it shows

1. **Plain evaluation.** `profit(plain, line, flags, knobs)` is the simulation on ordinary numbers.
2. **The gradient with respect to every minute's decision.** The pause flag of each station in each
   minute (1 440 of them) is relaxed from {0, 1} to a real number; the model is smooth in it
   (sigmoids for starving and blocking, a smoothed floor on wear), so one forward pass recording the
   tape and one backward sweep give d(profit)/d(pause) for every station-minute at once.
3. **First-order estimates of every move.** A move changes some flags from 0 to 1 and others from 1
   to 0; the gradient dotted with that change estimates its effect on profit, for all 150 candidate
   moves, from the one gradient. The estimates are not exact (the problem is far from linear), but
   they rank well: at the start, all of the top 20 improve profit, against 61 of the 150 overall.
4. **Local search in gradient order versus random order.** First-improvement local search evaluates
   candidates one by one and takes the first that improves profit. In gradient order it recomputes
   the gradient after each accepted move and tries candidates best-estimate first. Work is counted
   in evaluation equivalents: each full evaluation counts 1, and each gradient pass counts what it
   costs against a plain evaluation, which the example measures by timing both. That makes the work
   figures vary a little from run to run and machine to machine; the searches themselves are
   deterministic.
5. **Forward mode** gives the sensitivity of profit to one parameter in a single pass: to the price
   (which, profit being linear in it, is the value-weighted number of good parts sold), and to the
   rate at which the stations wear.

## Output

```text
A line of 3 stations over a 480-minute shift;
6 maintenance windows, 120 minutes in all.

1. Profit of the starting schedule (plain evaluation): 1202.86
2. One tape pass: profit 1202.86 and 1440 partial derivatives.
   station 0: d(profit)/d(pause) ranges -0.62 .. 8.95, least harmful minute to stop: 249
   station 1: d(profit)/d(pause) ranges -0.79 .. 11.90, least harmful minute to stop: 59
   station 2: d(profit)/d(pause) ranges -0.61 .. 12.58, least harmful minute to stop: 79

3. 150 candidate moves, ranked by their first-order estimate. The top five:
   shift window 0 by +5 min              estimated   27.92   actual   22.00
   move 3 min from window 3 to window 2  estimated   25.08   actual   16.38
   move 3 min from window 1 to window 2  estimated   23.72   actual   14.25
   shift window 0 by +4 min              estimated   22.03   actual   18.08
   move 3 min from window 3 to window 4  estimated   19.40   actual   21.00
   20 of the top 20 improve profit; 61 of all 150 do.

4. First-improvement local search from the same start. Work is in evaluation equivalents:
   full evaluations, plus 2.5 for each gradient pass (its cost here, measured). The best optimum
   any search found is 1741.50; the columns give the work to reach a share of its gain, then each
   search's own optimum, then all the work, the final round included.
   order              optimum  moves      50%      75%      85%      90% own opt.   in all
   gradient           1715.43    118       51       95      279     1365     1483     1591
   random (seed 1)    1696.37    193      135      408      606      948     1115     1230
   random (seed 2)    1729.84    170       65      263      423      602     1076     1196
   random (seed 3)    1726.45    199       79      273      391      684     1309     1434
   random (seed 4)    1741.50    216       64      182      469      659     1275     1410
   random (seed 5)    1726.09    173       76      268      446      654     1024     1144
   Gradient order against the median random search:
     50%       faster (51 against 76)
     75%       faster (95 against 268)
     85%       faster (279 against 446)
     90%       slower (1365 against 659)
     own opt.  slower (1483 against 1115)
     in all    slower (1591 against 1230)
   optimised windows (gradient order):
     station 0: minutes 124-152 (29 min)
     station 0: minutes 247-251 (5 min)
     station 1: minutes 106-140 (35 min)
     station 1: minutes 246-250 (5 min)
     station 2: minutes 209-249 (41 min)
     station 2: minutes 400-404 (5 min)

5. Forward mode, one pass each, at the optimised schedule:
   d(profit)/d(price)    = 221.43  (good parts sold, weighted by what each fetched)
   d(profit)/d(wearRate) = -414103  (so wearing 0.001 faster per minute costs 414.10)
```

## Reading the result

Measured against the best optimum any search found, and counting the gradient passes, the
gradient order is **faster early and slower late**. In the run above (a gradient pass measured at
2.5 plain evaluations) it reached half of the attainable gain in 51 evaluation equivalents, three
quarters in 95 and 85% in 279, against a median of 76, 268 and 446 for the random searches. It
needed 1365 to reach 90% (median 659), 1483 to reach its own optimum (median 1115), and 1591 in all
(median 1230): **it was costlier overall**, and its optimum (1715.43) was worse than four of the
five random searches'. Those last three comparisons hold whatever a gradient pass costs: counting
full evaluations alone, the gradient order still needs more.

Early on, every step has a clear direction and the first-order estimate ranks the moves well. Near
an optimum, every remaining move is small compared with how strongly the minutes of one window
interact (the yield in a window's later minutes depends on the wear its earlier minutes removed),
so the estimate is dominated by what it leaves out; the gradient order then still pays for a tape
pass per step and tries candidates in an order no better than chance. First-improvement search over
a small neighbourhood guarantees only a local optimum, which is why the searches end in different
places.

That is the honest shape of first-order information on a discrete problem: a good ranking while
there is a clear direction to go, and a cost once only curvature is left. A practical search would
use it for the first phase and switch to something else (random order, larger moves, or exact
evaluation of every candidate) for the last.

An earlier version of this example offered large moves as well (shifts of up to 30 minutes). With
those, gradient order did worse than random throughout: the estimate of a large shift grows
linearly with its size, but its real effect does not, so the ranking put the most overconfident
moves first. Keeping the candidate moves small enough for a first-order estimate to mean
something is part of using one.

## Tests

[test/example.test.ts](../../test/example.test.ts) runs the whole walkthrough on a two-station,
60-minute line, checks the gradient of profit with respect to the relaxed flags and both knobs
against central finite differences, and checks the forward-mode price sensitivity against the
difference one unit of price makes.
