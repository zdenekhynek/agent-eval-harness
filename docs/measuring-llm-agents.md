# Measuring LLM agents: which prompting strategy actually makes them smarter?

*Draft. Code, data and an interactive dashboard:
[github.com/zdenekhynek/agent-eval-harness](https://github.com/zdenekhynek/agent-eval-harness) ·
[live dashboard](https://zdenekhynek.github.io/agent-eval-harness/).*

Every week there is a new prompting trick that supposedly makes agents smarter:
think step by step, show examples, reflect on your mistakes, sample several
answers and vote. Most claims come with a single number from a single run. I
wanted to know two things: does the trick actually help, and how would I know?

The second question turned out to be the more interesting one. Before getting
a trustworthy answer to the first, I found three ways my own measurements were
wrong — each one invisible in a table of point estimates.

## The setup

The agents play small gridworlds, reimplemented from
[MiniGrid](https://minigrid.farama.org/) so the harness has no dependencies:

- **Empty**: walk to the goal. Pure spatial navigation.
- **LavaGap**: cross a lava column through its one gap. Stepping on lava ends
  the episode with zero reward, so it tests whether an agent reads the map
  before it moves.
- **DoorKey**: pick up the key, unlock the door, then reach the goal. Planning
  across subgoals.

Each turn the model gets a text map, its position and facing, and its last few
actions, and answers by calling an `act` tool. Movement is turn-then-move, so
"the goal is to my right" is not the same as `right`: you turn first, then go
`forward`. That turns out to matter.

Every number is scored against two baselines on the same tasks: a **random**
agent (score 0) and a **scripted** breadth-first solver that reads the true
state (score 1). Tasks then share one scale, and "0.6" means 60% of the way
from flailing to optimal.

## First result: reasoning matters — for the smaller model

Ten held-out seeds per task, a 40-step cap, two models, $3.33 in API costs:

| Agent | Normalised score, 95% CI | Success |
|---|---|---|
| ReAct + thinking · Sonnet 5 | 1.00 [0.99, 1.00] | 100% |
| ReAct · Sonnet 5 | 0.99 [0.98, 1.00] | 100% |
| Zero-shot · Sonnet 5 | 0.99 [0.98, 1.00] | 100% |
| ReAct · Haiku 4.5 | 0.96 [0.93, 0.98] | 100% |
| Zero-shot · Haiku 4.5 | 0.66 [0.52, 0.80] | 70% |

On Haiku 4.5, asking the model to reason in a few sentences before acting
(ReAct) lifts the score by **+0.30 [+0.15, +0.45]**. That interval comes from a
paired comparison: both agents played the same ten layouts, so the bootstrap
runs on per-seed differences, which is far tighter than comparing two separate
intervals.

The gain sits exactly where the tasks need thought. On Empty, ReAct and
zero-shot are indistinguishable (+0.02 [−0.01, +0.05]). On LavaGap it is
+0.44 [+0.11, +0.78], on DoorKey +0.43 [+0.13, +0.73]. Zero-shot Haiku walked
into lava three times in ten LavaGap episodes, once on its very first step, with
`Directly ahead: '~'` written in its observation. It also ran out of steps on
four of ten DoorKeys. ReAct Haiku never failed.

Sonnet 5 solved every seed with every strategy. These 5×5 rooms are too easy to
separate strong models, which is itself a result worth stating plainly: a
benchmark that every candidate saturates tells you nothing about them.

## Three ways the measurement was wrong

### 1. A hidden reasoning channel

My first ReAct run on Sonnet 5 used 571 output tokens on a step whose visible
reasoning was about forty words. The response contained a `thinking` block
with empty text: Sonnet 5 thinks privately by default, and the API returns the
block without its contents.

Zero-shot forces the tool call, which leaves no room for that thinking. So
"ReAct vs zero-shot" was really "visible reasoning **plus hidden thinking** vs
neither". Any difference would have been credited to the wrong cause.

The fix is to make thinking an explicit variable. Every strategy now sends
`thinking: disabled` unless its name says otherwise (`react+think`), and the
prompt fingerprint that guards a run includes the setting.

### 2. Anchors computed under different rules

Normalising needs reference returns for the random and scripted agents. I
computed them once, without a step limit. The runs used `--max-steps 40` to cap
cost. A capped random agent rarely stumbles onto the goal, so its scores in the
run were far below the reference, and random scored **−0.17** against its own
baseline.

Anchors have to be computed under the same rules as the episodes they
normalise. With references taken under the run's step cap, random scores
**−0.01 [−0.16, 0.14]**, as it should.

### 3. Significant is not the same as meaningful

Strong agents are nearly deterministic. When Sonnet 5 ReAct and zero-shot both
solve every seed, one taking a single extra step on one seed gives a
confidence interval like [+0.001, +0.009]: it excludes zero, so a naive test
calls ReAct "better".

That difference is real and meaningless. The harness now requires a win to
clear both bars: the interval excludes zero **and** the difference is at least
0.05, five percent of the gap between random and optimal. Smaller but real
differences are reported as *marginal*, and the dashboard says so.

## What the transcripts show

Scores say *whether* an agent failed; transcripts say *why*. The dashboard
replays every episode step by step, redrawing the grid from the text the model
actually saw.

A typical Haiku detour: facing south with a wall ahead and the goal to its
east, it reasons, correctly, that it needs to face east, and then turns
`right`. From south, right is west. It turns right again, now facing north,
walks back to the corner it started from and circles from there: 38 steps on a
task the scripted solver finishes in about six.
The action description says which way each turn rotates; the model still
confuses "the goal is on my right" with "turn right".

This is why the few-shot strategy's examples are about rotation, keys, doors
and lava rather than whole solved episodes: they target the specific mistakes
the transcripts show.

## What's next

- **Harder tasks.** I added MiniGrid's LavaCrossing: a 9×9 room crossed by up
  to three lava rivers, each with one opening. In a first single-seed look,
  Sonnet 5 zero-shot ran out of steps on two rivers where ReAct solved it in
  21. One seed proves nothing; the full run will.
- **More strategies.** Few-shot examples, Reflexion-style self-critique when
  the agent revisits a situation, and self-consistency voting over three
  samples are implemented and tested. The next study runs all of them on
  both models over the harder tasks.

## Takeaways for anyone evaluating agents

1. **Pair your comparisons.** Run every agent on the same seeds and test the
   per-seed differences.
2. **Control every channel of reasoning.** Defaults change between models;
   set them explicitly and record them with the run.
3. **Score against anchors computed under the same rules.**
4. **Decide in advance what difference matters.** Statistical significance
   alone will crown noise once your agents get good.
5. **Read the transcripts.** The numbers tell you where to look.
