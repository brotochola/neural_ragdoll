// Every tuning number lives on this object. The sim, the reward, and the trainer read these names.

const physicsHz = 60;
const partCount = 15; // Lower torso, chest, head, and the limbs. The tail is extra.
const jointSenseCount = 28; // 14 joint angles and 14 joint rates. The extra joint is the waist.
const bodyRayCount = 29; // Short rays from torso, chest, arms, hand tips, legs, soles, toes, and the tail except up.
const headRadius = 0.12;
const headAxis = 0.024;
const neckY = 1.56;
const headCenterY = neckY + headRadius + headAxis;
const headForwardRayCount = 4; // Most head rays look forward.
const headBackRayCount = 2; // A few look behind the head.
const headRayCount = headForwardRayCount + headBackRayCount; // Observation width of the head fan. Changing this needs a fresh network.
const headRayMeters = 2; // No head ray looks farther than this.
const episodeSeconds = 6; // One scored individual, unless the head hits first.
const targetCount = 5; // Head, handR, handL, footR, footL.
const targetStride = 5; // use, dx, dy, sin of the angle error, cos of the angle error. Each is its own input.
const targetInputCount = targetCount * targetStride;
const sensedBeforeRays = jointSenseCount + 4 + 2 + 4; // Joints, torso/head sin/cos, two heights, four mass values.
const targetInputIndex = sensedBeforeRays + bodyRayCount + headRayCount;
const sensedInputCount = targetInputIndex + targetInputCount + 1; // The last column is head contact.
const rewardRowCount = targetCount * 2; // Position and rotation for each target.
const rewardTotalNow = false;
const rewardTotalPrevious = false;
const rewardTotalNext = false;
const rewardTotalInputCount =
  Number(rewardTotalNow) +
  Number(rewardTotalPrevious) +
  Number(rewardTotalNext);

export const config = {
  physics: {
    hz: physicsHz, // Steps per simulated second.
    timeStep: 1 / physicsHz, // Seconds advanced by one physics step.
    solverSubsteps: 2, // Box2D solver iterations per step. 3 was steadier; 2 is the fast setting.
  },
  cadence: {
    actionEveryNFrames: 6, // Frames an action is held before the net picks a new one. 5 = 12 decisions/s (was 10 = 6/s, slow for balance).
    rewardEveryNFrames: 10, // Frames between score samples. At 60 Hz this is 6 samples a second.
    rewardFrameOffset: 1, // Sample the score this many frames after a new action, not on the action frame.
  },
  net: {
    hidden1Ratio: 0.5, // First hidden layer as a fraction of inputCount. Rounded to a whole number of units.
    hidden2Ratio: 0.25, // Second hidden layer, also a fraction of inputCount. 0 turns it off.
    feedbackUnits: 0, // Nonlinear summary of the previous action. It enters the recurrent layer only.
    jointCount: 14, // Motor joints, and the width of the action vector. Waist, neck, and six joints on each side.
    sensedInputCount, // Observation length before optional reward-total columns.
    targetInputIndex, // First of the target columns. Five targets, five numbers each: use, dx, dy, error sin, error cos.
    targetInputCount,
    targetCount,
    headContactInputIndex: sensedInputCount - 1, // 1 when Box2D says the head is touching.
    partCount, // Dynamic body parts. The tail is extra and is not one of them.
    rewardRowCount, // Score rows shown on screen. They are not copied into the observation.
    rewardTotalNow, // Append the reward total from this decision.
    rewardTotalPrevious, // Append the reward total from the previous decision, one action interval ago.
    rewardTotalNext, // Append the reward total estimated one physics frame ahead.
    inputCount: sensedInputCount + rewardTotalInputCount, // Full observation length.
    freshInputExplorationScale: 0.04, // Noise on columns after sensedInputCount (reward totals, when on).
    hiddenLeak: 0.6, // Fraction of the previous hidden state kept each step.
  },
  body: {
    platformPeriodSeconds: 8, // Kept on the task header. The floor does not tilt.
    touchGapMeters: 0.005, // Clearance below this counts as touching the floor.
    standingHeadMeters: 1.45, // Head height required to count as standing.
    standingTorsoAngleRadians: (15 * Math.PI) / 180, // Max absolute torso tilt that still counts as standing.
    standingStanceMinMeters: 0.05, // Narrowest stance that still counts as standing.
    standingStanceMaxMeters: 0.9, // Widest stance that still counts as standing.
    headRadius, // Head capsule radius.
    headAxis, // Half the distance between the head capsule's circle centers.
    headHeightMeters: headCenterY, // Head center of the standing blueprint. The head-height input uses this.
    torsoHeightMeters: 1.14, // Lower-torso center of the standing blueprint.
    waistForwardRadians: (35 * Math.PI) / 180, // Chest flexes this far toward the toes.
    waistBackRadians: (20 * Math.PI) / 180, // Chest extends this far backward. Less than the forward bend.
    bodyRayMeters: 0.35, // Short rays from the body. 1 means nothing is within this distance of that face.
    targetSpanMeters: 2, // Target distance at which the position reward reaches its floor.
  },
  vision: {
    headRayMeters, // Cap on every head ray, in meters.
    headForwardCount: headForwardRayCount, // Rays across the face. Local +X is forward.
    headForwardArcRadians: Math.PI, // That fan covers the front half, centered on the face.
    headBackCount: headBackRayCount, // Rays behind the head.
    headBackArcRadians: Math.PI / 2, // Those cover a narrower arc, centered on the back of the head.
  },
  // Standing blueprint. Head starts on. The other four start off, at the part they belong to.
  targets: [
    { id: "head", part: "head", on: true, x: 0, y: headCenterY, angle: 0 },
    { id: "handR", part: "handRight", on: false, x: 0, y: 0.91, angle: 0 },
    { id: "handL", part: "handLeft", on: false, x: 0, y: 0.91, angle: 0 },
    { id: "footR", part: "footRight", on: false, x: 0.06, y: 0.04, angle: 0 },
    { id: "footL", part: "footLeft", on: false, x: 0.06, y: 0.04, angle: 0 },
  ],
  reward: {
    targetPosition: 1, // +this on the target, -this at targetSpanMeters. Only while that target is on.
    targetRotation: 1, // Cosine of the angle error. +this when aligned, -this when flipped. Only while that target is on.
    headTouch: 2, // Subtracted when a training episode's head touches. The rest of the score stays, so a late brush still ranks above an early dive.
    motion: 0.02, // Subtracted times the mean |motor speed| / motorSpeedLimit. A held pose costs nothing. A joint at the speed limit costs this.
  },
  curriculum: {
    // Rows the Train menu and the headless run walk through. Each row lasts `generations`.
    // start "up" is standing. "mixed" and "down" are a random tilt and a random height.
    phases: [
      {
        id: 1,
        name: "Stand",
        generations: 2000,
        start: "mixed",
        falls: false, // Turn back on once the stand eval passes.
        balls: false, // Turn back on once the stand eval passes.
        goal: false,
      },
      {
        id: 2,
        name: "Get up",
        generations: 4000,
        start: "down",
        falls: false,
        balls: false,
        goal: false,
      },
      {
        id: 3,
        name: "Walk",
        generations: 4000,
        start: "up",
        falls: false,
        balls: false,
        goal: true,
      },
    ],
    phase: 1, // Active phase id. setTrainingPhase writes this.
    byPhases: false, // When true, a run stays on each phase for its generation budget, then advances.
  },
  scale: {
    rewardInput: 8, // A reward row is divided by this before it enters the net.
  },
  start: {
    fallDropChance: 0.25, // Chance an upright start is instead dropped from height, when the phase allows falls.
    fallDropMinMeters: 1.2, // Lowest drop height.
    fallDropMaxMeters: 2.2, // Highest drop height.
    ballChance: 0.35, // Chance a ball is thrown, when the phase allows balls.
    ballThrowFrame: 180, // Frame on which the ball is spawned.
    ballSpeed: 8, // Horizontal speed of a thrown ball, m/s.
    ballStartMeters: 2.2, // How far to the side of the torso a ball starts.
    uprightDropMeters: 0.12, // Gap under the feet of an upright start that is not a fall.
    uprightTiltRadians: 0.08, // Max tilt of an upright start.
    targetJitterMeters: 0.06, // A training mark sits at most this far from the part it belongs to, at the start of the episode.
    targetJitterRadians: 0.15, // A training mark turns at most this far from how that part is facing at the start.
    downTiltMinRadians: 1.15, // Minimum tilt of a face-up or face-down start.
    downTiltSpanRadians: 0.55, // Extra random tilt added past the minimum.
    downDropMeters: 0.05, // Gap under a face-up or face-down start before it settles.
    downSettleFrames: 40, // Physics steps run before the episode clock, so a down start reaches the floor.
  },
  search: {
    explorationScale: 0.06, // Target-column noise never drops below this.
    explorationFloor: 0.12, // Starting amplitude of the decaying target noise. The used scale is the max of this decay and explorationScale.
    explorationHalfLifeGenerations: 4000, // Generations for the decaying noise to halve.
    stagnationGenerations: 1000, // Plateau length before the large noise starts. One new best removes this many generations from the count.
    stagnationBoost: 3, // Noise multiplier at the first plateau, then again after each further stretch of the same length.
    stagnationBoostCap: 9, // Two boosts, then sigmaCap already clips limb noise.
    sigmaCap: 0.4, // One weight's noise never exceeds this. At the plateau cap every scout stops here, near a trained weight.
    recordWindow: 16, // Generations in the ending average that sets a record.
    recordMargin: 0.2, // That average must clear the previous record by this much.
    bestEverPairs: 2, // Extra antithetic pairs copied from Best ever with the small noise. 0 turns them off. They are not scouts and they do not enter the Adam step.
    bestEverAdoptMargin: 0.25, // During a plateau, a small-noise child replaces the parent when it beats its twin and the parent average by this much. A Best ever copy only has to beat the parent average. Scouts never become the parent.
    stagnationMutateCap: 0, // Share of pairs that receive the large noise once a plateau starts. 0 keeps every pair on the small noise.
    limbExplorationScale: 0.05, // Noise on shoulder, elbow, hip, knee, and ankle outputs.
    bodyExplorationScale: 0.02, // Noise on hidden weights. Small, so balance can shift without a rewrite.
    learningRate: 0.002, // Adam step size at generation 0. A tenth of the hidden-weight noise.
    learningRateFloor: 0.0005, // Adam step size never drops below this.
    learningRateHalfLifeGenerations: 8000, // Generations for the Adam step size to halve.
    revertAfterGenerations: 400, // A plateau this long copies Best ever back onto the parent and clears Adam. 0 leaves the parent where it wandered.
    trainableJointIndexes: [0, 2, 3, 5, 6, 7, 8, 9, 11, 12, 13], // Waist, shoulders, elbows, hips, knees, ankles. Neck and wrists stay put.
    adamBeta1: 0.9, // Adam first-moment decay.
    adamBeta2: 0.999, // Adam second-moment decay.
    adamEpsilon: 1e-8, // Added under the Adam denominator.
    hiddenInitScale: 0.12, // Fresh hidden weights are a gaussian times this.
    outputInitScale: 0.04, // Fresh joint-output weights are a gaussian times this.
    randomHiddenScale: 0.26, // Random-weights button. Gaussian scale of a trained hidden block.
    randomOutputScale: 0.22, // Random-weights button. Gaussian scale of a trained joint-output block.
    goalCloseEnoughMeters: -1, // Elite replace also needs the best child at least this close to the marker.
    eliteMissingGapMeters: 9, // Gap assumed when a score has no marker distance.
    headImpactQuiet: 1.5, // A stand counts as quiet when the peak head-down speed is below this, in m/s.
  },
  episode: {
    seconds: episodeSeconds, // How long one scored individual runs. A head touch subtracts reward.headTouch and the episode keeps going.
    frames: episodeSeconds * physicsHz, // Physics steps in that run. 6 s at 60 Hz is 360. The doll on screen is not cut by this.
    startWindowSeconds: 0.25, // Mean reward over the first this many seconds. A drop from here to the ending is subtracted from the fitness.
    endWindowSeconds: 3, // Mean reward over the last this many seconds. The fitness is this, minus any drop from the start window.
    startTiltJitterRadians: 0.12, // Extra tilt added on top of a situation.
    startShiftMeters: 0.4, // Extra horizontal shift of a situation.
    startDropJitterMeters: 0.06, // Extra drop height added on top of a situation.
    motorGain: 12, // How hard a joint motor chases the action target.
    motorSpeedLimit: 10, // Max joint motor speed, radians per second.
    goalNearMeters: 1.2, // Closest random marker distance.
    goalFarExtraMeters: 1, // Extra meters added at random past the closest marker.
  },
  workers: {
    browserMin: 1, // Fewest browser workers.
    browserMax: 8, // Most browser workers.
    browserReserve: 1, // CPU cores left unused when sizing the browser pool.
    browserHardwareFallback: 4, // Core count assumed when the browser does not report one.
    browserPairMin: 48, // Fewest antithetic pairs in a browser generation. Was 4: too few samples for ~1100 weights.
    headlessPairs: 48, // Antithetic pairs in a headless generation.
    headlessWorkerMin: 2, // Fewest headless workers.
    headlessWorkerMax: 12, // Most headless workers.
    headlessWorkerReserve: 2, // CPU cores left unused when sizing the headless pool.
  },
  session: {
    everyGenerations: 20, // Browser weights are written to IndexedDB this often, and also on pause, hide, and restart.
  },
  eval: {
    headlessN: 40, // Held-out episodes per kind for an official headless eval.
    quickN: 10, // Episodes per kind in a quick eval.
    quickEvery: 50, // Generations between quick evals.
    fullMultiple: 4, // A full eval runs every this many quick evals.
    seedBestCount: 8, // Episodes averaged to seed bestEverScore when a phase starts.
    officialMinN: 20, // Pass/fail is official only when the eval count is at least this.
    nearPassSlack: 0.15, // A rate this close to the pass rate still triggers a full eval.
    passRate: {
      stand: 0.8,
      ball: 0.7,
      drop: 0.65,
      supine: 0.6,
      prone: 0.6,
      walk: 0.6,
    }, // Fraction of held-out episodes that must pass each kind.
    walkGoalMeters: 2, // Marker distance used by the walk eval.
    walkClosingMeters: 0.5, // A walk passes when the gap to the marker shrinks by at least this.
    seed: 424242, // Seed of the held-out eval situations.
    shiftMeters: 0.15, // Horizontal jitter of a non-walk eval start.
  },
  partIndex: {
    // Body part indexes. Same order the sim builds them.
    torso: 0, // Lower torso, the pelvis.
    chest: 1,
    head: 2,
    upperArmLeft: 3,
    forearmLeft: 4,
    handLeft: 5,
    thighLeft: 6,
    shinLeft: 7,
    footLeft: 8,
    upperArmRight: 9,
    forearmRight: 10,
    handRight: 11,
    thighRight: 12,
    shinRight: 13,
    footRight: 14,
    hip: 15, // Tail circle welded under the pelvis. Not one of the speed inputs.
  },
};

export function setTrainingPhase(phase) {
  const id = phase | 0;
  config.curriculum.phase = config.curriculum.phases.some(
    (item) => item.id === id,
  )
    ? id
    : config.curriculum.phases[0].id;
}

export function setTrainingByPhases(on) {
  config.curriculum.byPhases = !!on;
}

export function activeTrainingPhase() {
  return (
    config.curriculum.phases.find(
      (item) => item.id === config.curriculum.phase,
    ) || config.curriculum.phases[0]
  );
}
