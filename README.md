# Neural ragdoll

A 2D ragdoll and one Elman network, trained in the browser or headless. The doll is scored
part by part for standing, and for reaching marks on the floor.

## Open it

```
node serve.js
```

Then open http://localhost:8765/. Box2D is loaded from a CDN build (`PhaserBox2D.min.js`).
Headless training uses the vendored copy and needs no server.

## The reward is only the targets

The fitness is the mean of the ten target rows — five positions, five rotations — sampled across
the **whole episode**. That is all of it.

There is no `upright`, no `headTouch`, no `alive`, no `fail`, no `motion`, no `step` term.
They are all zero in `config.reward` and none of them is read by `fitnessValue`.

**Standing is not a term. It falls out of the frozen marks.** Head and hand marks are planted
**once**, at reset, using the midpoint of the feet only to capture the standing torso height —
and then they never move again for the rest of the episode. The mark lives in the world, so the
torso has to stay up to reach it. Tip over and the head falls away from its own frozen mark, and
the head row charges you for it:

```
standing        reward  7.92   head height 1.63
lying on back   reward  4.74   head height 0.12
```

`check-collapse.js` measures exactly that, and fails the build if lying ever scores higher.

The marks being frozen is also why dragging one in the sandbox behaves: it stays where you put
it. Before, `plantBodyTargets` rewrote `x = feetMid + lx` on **every frame**, so the mark chased
your feet and snapped back while you were dragging it.

### The feet are commanded only when the level says so

`fillStandFeet` used to set the foot stance by writing `±stanceMeters` into each foot's `dx`. But
`dx` is the *step command* field, so the stance silently became a command. At skill 1 — the level
called **stand** — the right foot was being asked for a 12 cm step in the same episode it was
asked not to fall over, and the feet are 6 of the 12 points.

The stance now goes in the pose (`options.stance`, applied in `resetPose`), where it belongs. The
foot mark is planted where the foot actually is, plus the commanded `dx` only when that level
commands feet. So:

- skill 1 — the foot row means "don't move your foot"
- skill 4 — the foot row means "take this step"

### An anchoring experiment that lost, and why

Anchoring foot marks to the hip instead of the world *does* make the row measure the step cleanly
(drift cancels exactly). It also looks better on `check-collapse`, which compares **held poses**:
7.31 standing vs −4.99 lying, a 10.31 gap where world-anchoring gave 3.18.

It loses in training, badly: 4/20 at gen 300 against 9/20, then down to 0/20 and stuck there.
While the torso goes down the hip goes down *with the feet*, so the mark follows and the foot
banks its points without the body being up at all. Gifting 6 of 12 points during a fall is
gifting half the reward.

Worth recording because `check-collapse` cannot see it — it holds the pose still, training doesn't.
The residual drift (0.298 m against a 0.089 m command, from `check-foot-command.js`) is not an
anchoring problem at all. It is what "the net cannot stand yet" looks like. Once it stands, drift
goes to zero and the frozen world mark measures the command by itself. Skill 1 before skill 4.

The starting pose enters through one number only, `reward.startHoldShare`: **how much of what was
lost between the start and the end is charged.** It uses the same ten rows. It charges the drop
and nothing else — improving during the episode is free.

It used to be `(1-share)·life + share·min(start, life)`. Since `min` is always ≤ `life`, that
charged whoever *improved*: taking a foot step raises the episode average, and the fitness was
left halfway to not having taken it. For the foot skill that was half the signal, inverted.
Now it is `life − share·max(0, start − life)`. `check-reward.js` pins all four cases.

### The position row shape

```
row = weight * (1 - 2r / sqrt(1 + r²))        r = distance / span
```

This one line has been wrong twice, and both failures were invisible until measured:

- **Linear `1 - 2r`**, floored at a span. A fall is ~1.5 m and a span is 0.25 m, so every
  collapsed pose scored identically. A plateau: no gradient back up.
- **`1 / (1 + r²)`**, floored at 0 instead of at `-weight`. No plateau, but now lying down is
  worth ≈0 rather than ≈`-weight`, and the feet rows are worth 6 of the 12 points. Measured
  consequence: standing 10.92 vs lying 8.00 — a 24% gap where the linear version had 100%.
  The fix for the plateau had eaten the reward for standing.

The current form has all three properties: derivative `-2·weight/span` **at the mark**, identical
to the linear row, so the fine precision where learning actually happens is untouched;
`-weight` in the far field, so falling still costs the whole weight; and strictly decreasing
(`d/dr [r/√(1+r²)] = 1/(1+r²)^{3/2} > 0`), so there is no plateau anywhere.

## The parent was frozen for 400 generations

A 400-generation skill-1 run went `0/20 → 3/20 → 7/20 → 1/20` on the standing gate while the
record climbed the whole time. That is not a reward problem. That is a search with no elitism.

`tell()` promotes a child to parent only if it beats `eliteScore` **and** passes a goal gate:

```js
(scores[i].gap ?? 9) < search.goalCloseEnoughMeters    // goalCloseEnoughMeters was -1
```

A distance is never less than `-1`, so the gate was false for every possible `gap`, including the
`?? 9` fallback. The parent could not be replaced **ever**. The only remaining path was
`adoptAnchor`, which needs `stagnationGenerations` (1000) of plateau — it never fired in 400
generations. The population mean was rising purely from the Adam step at `net.js:440`, not from
selection, and a parent that never improves is a random walk. That is exactly the
7/20 → 1/20 shape.

`net.js` now skips the gate when the situation has no `goalX` — all five target levels have no
goal, there is nothing to be close to — and `goalCloseEnoughMeters` is back to `0.45`, a real
distance, for the situations that do have one. The search reports `parentMoves`; the trainer logs
every parent change and warns explicitly if the parent never moves. **Read that number before
blaming the reward** — it distinguishes "the reward gives no signal" from "the search is not
keeping what it found".

## Network

109 inputs → 50 tanh → 25 leaky-tanh recurrent (leak 0.6) → 14 tanh joint outputs.

- Joint angles and rates (28), torso/head angle, heights, CoM position and velocity.
- **x/y velocity of each of the five target parts** (10). The net used to see position error
  but not how fast the part was closing on it, so a hand flying past its mark and one drifting
  toward it looked identical.
- 35 raycast clearances, 25 target columns, head contact.
- No feedback of the previous action (`feedbackUnits: 0`).

`net.skipTargetInputs` adds a direct path from the target columns to the joint outputs. It is
**off**: it was tried, and it took skill 1 from 0.95 at gen 160 to 0.45 at gen 480. The wiring
is kept behind the flag with that note.

Head, handR, handL, footR, footL. A command is **anatomical neutral plus a random offset**
over the workspace the joint can actually reach, sampled *independently of where the doll
starts*. That independence is the whole point: a command derived from the start pose is
satisfied by standing still, and the net learns to hold a pose instead of obeying one.

- Head and hands ride the torso frame. Their marks follow the pelvis, so the row measures
  posture, and the error that enters the network is rotated into the torso frame too.
- Their **angle** is world level and does *not* follow the torso: level head, level hands.
- Feet are commanded in world axes, on the floor.

Per target in `config.targets`: `span` (where the position reward bottoms out), `weight`,
`ride`, and the command workspace `cmdX` / `cmdY` / `cmdAngle`. `cmdArc` means "derive the
position from the commanded angle", which the head needs: its capsule pivots 0.144 m from the
neck, so its body-frame position can only ever sit on a 14 cm arc.

`config.commands` says which group gets a real command at each level:
**1 stand, 2 head, 3 hands, 4 feet, 5 all five.** A level passes when the doll is standing and
every target of that group is inside tolerance — 5 cm and 0.15 rad for head and hands,
6 cm and 0.3 rad for feet. `config.targetGains` scales any group between 0 and 1.

## Checks

```
npm run check
```

Four gates, none of which need a trained network:

- `check-reward.js` — reward arithmetic, per-metre sensitivity of each target, and the four cases
  of the start blend.
- `check-reach.js` — every sampled command is inside the joint's reach sphere. Arms and legs
  start *fully extended*, so a naive workspace asks for the impossible.
- `check-obey.js` — **a net that ignores the commands must not pass a gate.** This is the one
  that catches a bar which measures nothing: if standing still clears it, the bar is worthless.
- `check-collapse.js` — **the target rows alone must prefer standing to lying down.** Without
  frozen marks the net learns to fall over.
- `check-foot-command.js` — **the foot row must measure the command, not the body's drift.**
  Decomposes the foot error into command / floor offset / drift and fails when the drift is more
  than twice the command. `--net <checkpoint>` runs it against a trained net instead of a falling
  one.

All four guard against silent garbage: `check-collapse` asserts no NaN (a short weight array
makes the net read out of bounds, and `NaN.toFixed(2)` prints `"0.00"`, which looks like a
passing measurement) and asserts the poses actually differ.

## The episode-length ramp

Training a 6-second "do not fall over" from generation 0 asks the search for a 6-second balance
controller with no stepping stone. `config.skill.rampStartSeconds` (1.5 s) and
`rampGenerations` (400) start the scored episode short and stretch it to the full length.

The reward does not change. It is the same ten rows, sampled over a shorter episode. And
`evaluateSkill` always runs the **full** episode regardless of where the ramp is — otherwise the
gate would be approving a network that can only stand for a second and a half.

This is a curriculum, not a reward term. Nothing was added to the score.

## Checks

```
node check-reward.js         # row arithmetic, sensitivities, the start-blend rule
node check-reach.js          # every sampled command is inside the joint's reach sphere
node check-obey.js           # a net that ignores commands must not pass a gate
node check-collapse.js       # the target rows alone must prefer standing to lying
node check-foot-command.js   # the foot row must measure the command, not body drift
npm run check                # all five
```

`check-foot-command.js` only **fails** if the net stands in at least 60% of episodes. A net that
cannot stand has a body drift of ~0.28 m against a ~0.09 m command, and that number measures
skill 1, not the foot row. Failing the build on it would be blaming the reward for a search that
hasn't converged yet — so it warns instead, and says so.

## Headless training

```
node train-headless.js --fresh --skill 1 --minutes 90      # from zero, walk the ladder
node train-headless.js --from backups/g150_s5p191_phase_xxx.json --skill 2
node train-headless.js --gains body=0                      # feet only
node train-headless.js --gains hand=0.5                    # ramp a group in
node train-headless.js --target-mean 0.4                   # how hard the targets count
node train-headless.js --residual-search                   # old behaviour: freeze the main net
```

`config.skill.scriptedLegs` turns the hand-written gait back on. It overwrites
hip/knee/ankle on its own clock, so it cannot obey a foot command, and the command ladder runs
with it off.

### Resuming after a fitness change

`config.reward.version` is written into `train-state.json`. If it does not match the running
config, the stored `bestEverScore` is thrown away and re-seeded: the old number is on a
different scale, and the parent would never beat its own record. Bump that number whenever
the fitness formula changes.

## Is the net actually using a target?

```
node probe-targets.js --skill 2 --n 24
node probe-targets.js --random          # baseline with random weights
node probe-targets.js --oracle 6        # replace one joint with a P controller
```

`corrige` is the number that matters: the fraction of the commanded displacement the net
actually removes. 0% means it is ignoring the command. A big fitness drop when you mute a
target is *not* proof of aiming — zeroing a column the net was using as a phase cue is a
distribution shock, and the doll falls for unrelated reasons.

`--oracle` hands one joint a proportional controller on the target error, keeping the net for
everything else. It answers *"could any policy do this?"*. If the oracle corrects a lot and the
net corrects nothing, the task is learnable and the search is what is failing — which is the
diagnosis to make before touching the reward again.

## What you are watching in the browser

- **Parent** — the weights being updated. They move a little every generation.
- **Best this generation** — the individual that scored highest in the last generation.
- **Best ever** — the highest 20-generation average since this session started.

The Head / Feet / Hands boxes score the sandbox doll. Training places its own marks; the boxes
do not change training.

The numbers on each reward row are set in `config.targets` and `config.reward`.