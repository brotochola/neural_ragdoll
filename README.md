# Neural ragdoll

A 2D ragdoll and one Elman network, trained in the browser. The doll is scored part by part for standing, and for walking toward a marker on the floor.

## Open it

```
node serve.js
```

Then open http://localhost:8765/. Box2D is loaded from a CDN build (`PhaserBox2D.min.js`). There is no npm install.

The session is stored in the browser. Reload keeps the weights.

## What you are watching

- **Parent** — the weights being updated. They move a little every generation.
- **Best this generation** — the individual that scored highest in the last generation. Replaced every generation.
- **Best ever** — the highest score since this session started. Changes only when someone beats it.

## How a generation works

Physics runs at 60 Hz. Box2D solves each frame in 2 substeps.

Each agent picks a new action every 10 frames and holds it. The reward is sampled every 10 frames, 5 frames after the action. Agent 0 decides on frame 0, agent 1 on frame 1, and so on, so they do not all update on the same frame.

The fitness of an episode is the average of those samples.

## Reward rows

Each row is one measurement. Green adds, red subtracts. The large **Standing** bonus needs both feet down, nothing else touching, an upright torso, and a high head.

Angles are the cosine of that bone: +1 is vertical, 0 is on its side, -1 is upside down.

Head height and torso height stay at their floor value until the torso turns up, and they stay there if a hand is holding the body.

Goal distance and goal speed pay less while the doll is on the floor than while it is standing, so diving at the marker is not the best plan.

The numbers on each row are set in `config.js`.
