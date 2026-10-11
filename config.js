// Every tuning number lives on this object. The sim, the reward, and the trainer read these names.

const physicsHz = 60;
const partCount = 14;
const sensedInputCount = 80; // 26 joint, 4 sin/cos, 2 heights, 4 mass, 40 face rays, goal x/y, standing, head contact.
const rewardRowCount = 29;
const partSpeedInputCount = partCount * 3;
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
    solverSubsteps: 3, // Box2D solver iterations per step. 4 was stiffer; 2 is faster.
  },
  cadence: {
    actionEveryNFrames: 6, // Frames an action is held before the net picks a new one. 5 = 12 decisions/s (was 10 = 6/s, slow for balance).
    rewardEveryNFrames: 2, // Frames between score samples.
    rewardFrameOffset: 1, // Sample the score this many frames after a new action, not on the action frame.
  },
  net: {
    hidden1Ratio: 50 / 149, // First hidden layer as a fraction of inputCount. Rounded to a whole number of units.
    hidden2Ratio: 50 / 149, // Second hidden layer, also a fraction of inputCount. 0 turns it off.
    feedbackUnits: 8, // Nonlinear summary of the previous action. It enters the recurrent layer only.
    jointCount: 13, // Motor joints, and the width of the action vector.
    sensedInputCount, // Observation length before part speeds. Goal x is 76, goal y is 77, standing is 78, head contact is 79.
    goalInputIndex: 76, // Index of the goal-x observation. Goal y is the next index.
    standingInputIndex: 78, // Index of the binary standing observation.
    headContactInputIndex: 79, // Index of the head-contact observation. 1 when Box2D says the head is touching.
    partCount, // Dynamic body parts, including the hip.
    partSpeedInputCount, // Three numbers per part: vx, vy, spin.
    rewardRowCount, // Score rows shown on screen. They are not copied into the observation.
    rewardTotalNow, // Append the reward total from this decision.
    rewardTotalPrevious, // Append the reward total from the previous decision, one action interval ago.
    rewardTotalNext, // Append the reward total estimated one physics frame ahead.
    inputCount: sensedInputCount + partSpeedInputCount + rewardTotalInputCount, // Full observation length.
    freshInputExplorationScale: 0.04, // Noise on the part-speed columns. Less than the goal columns.
    hiddenLeak: 0.6, // Fraction of the previous hidden state kept each step.
  },
  body: {
    platformHalfWidth: 20, // Half the platform length, in meters.
    platformHalfThickness: 0.1, // Half the platform thickness, in meters.
    platformPeriodSeconds: 8, // Period of a tilting platform, when one is enabled.
    touchGapMeters: 0.005, // Clearance below this counts as touching the floor.
    comfortableStanceMeters: 0.29, // Stance width that scores the full stance comfort.
    standingHeadMeters: 1.45, // Head height required to count as standing.
    standingTorsoAngleRadians: (15 * Math.PI) / 180, // Max absolute torso tilt that still counts as standing.
    standingStanceMinMeters: 0.05, // Narrowest stance that still counts as standing.
    standingStanceMaxMeters: 0.9, // Widest stance that still counts as standing.
    headHeightMeters: 1.68, // Head height of the standing blueprint. The height reward and the head-height input use this.
    torsoHeightMeters: 1.26, // Torso height of the standing blueprint.
    handClearMeters: 0.15, // A hand this close to the floor is fully "on the floor" for the hand penalty.
    thighClearMeters: 0.35, // A thigh this close to the floor is fully "on the floor" for the thigh penalty.
    thighSpreadMinRadians: (10 * Math.PI) / 180, // Thighs closer than this are penalized once the torso is upright.
    thighSpreadMaxRadians: (40 * Math.PI) / 180, // Thighs wider than this are penalized once the torso is upright.
    balanceSpanMeters: 0.4, // Center-of-mass offset from the feet that spends the whole balance range.
    goalSpanMeters: 2.5, // Marker distance at which the goal-distance reward reaches zero.
    goalWalkSpeed: 0.5, // Horizontal speed, m/s, that scores the full goal-speed reward.
  },
  reward: {
    torsoAngle: 1, // Weight of the torso cosine. +1 of this when upright, -1 when upside down.
    headAngle: 0.75, // Weight of the head cosine.
    upperArmAngle: 0.12, // Weight of each upper-arm cosine. +1 when that arm is vertical.
    forearmAngle: 0.15, // Weight of each forearm cosine.
    thighAngle: 1.5, // Weight of each thigh cosine.
    shinAngle: 0.25, // Weight of each shin cosine.
    kneeBend: 2, // Per knee. Cost is this times (1 - cos of the bend) times how far the torso has turned up. Straight is free. A small bend is almost free. A right angle spends the whole weight.
    headHeight: 4, // Head-height weight. The floor value is -this; a full upright reach is +this.
    torsoHeight: 1.8, // Torso-height weight. Same shape as the head height.
    handOnFloor: 0.05, // Penalty weight while a hand is near the floor.
    handPark: 1.5, // Extra hand penalty once the torso is already upright.
    elbowOnFloor: 2, // Per elbow, within the hand clearance. More than a hand: a hand is 0.05, or 1.55 once the torso is upright.
    thighOnFloor: 0.5, // Penalty weight while a thigh is near the floor.
    tailOnFloor: 1.5, // Penalty weight while the tail is near the floor. Not gated by an upright torso.
    thighSpread: 4, // Wide thigh opening. A right angle between the thighs spends all of this, even before the torso is upright. Closer than the minimum still waits for an upright torso.
    footInAir: 1.5, // Penalty weight per foot off the floor, only while upright.
    balance: 1.6, // Weight of the center of mass against the midpoint of the feet.
    headImpact: 16.8, // Penalty weight while the head is touching and moving down. A fast hit can add up to twice this again. A corner counts.
    headOnFloor: 10, // Penalty while the head touches anything.
    headPark: 2, // Added on top of headOnFloor once the torso is upright. Same gate as the hands.
    standing: 5, // Standing bonus, split across flat soles, a clear body, an upright torso, and a high head.
    standingFlag: 3, // Extra bonus when isStanding is true: both feet, nothing else down, upright, high head, a real stance, and quiet.
    stanceComfort: 2, // Bonus for a stance near the comfortable width, scaled by how flat the soles are. A very wide stance subtracts.
    goalDistanceUp: 0, // Marker-closeness weight while on the feet. Phase 3 replaces this.
    goalDistanceDown: 0, // Marker-closeness weight while not on the feet.
    goalSpeedUp: 0, // Toward-marker speed weight while on the feet. Phase 3 replaces this.
    goalSpeedDown: 0, // Toward-marker speed weight while not on the feet.
    motion: 2, // Penalty weight for part speed and spin while the torso is upright.
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
    phase3GoalDistanceUp: 2, // Marker-closeness weight while on the feet, used only when the active phase has a goal.
    phase3GoalDistanceDown: 0.7, // Marker-closeness weight while down, in a goal phase.
    phase3GoalSpeedUp: 1.2, // Toward-marker speed weight while on the feet, in a goal phase.
    phase3GoalSpeedDown: 0.4, // Toward-marker speed weight while down, in a goal phase.
  },
  scale: {
    partSpeed: 3, // Part linear speed is divided by this before it enters the net.
    partSpin: 6, // Part angular speed is divided by this before it enters the net.
    rewardInput: 8, // A reward row is divided by this before it enters the net.
    motionFull: 12, // Sum of part speeds and spins that spends the whole motion penalty.
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
    downTiltMinRadians: 1.15, // Minimum tilt of a face-up or face-down start.
    downTiltSpanRadians: 0.55, // Extra random tilt added past the minimum.
    downDropMeters: 0.05, // Gap under a face-up or face-down start before it settles.
    downSettleFrames: 40, // Physics steps run before the episode clock, so a down start reaches the floor.
  },
  search: {
    explorationScale: 0.18, // Noise never drops below this.
    explorationFloor: 0.22, // Starting amplitude of the decaying noise. The used scale is the max of this decay and explorationScale.
    explorationHalfLifeGenerations: 6000, // Generations for the decaying noise to halve.
    stagnationGenerations: 5, // Plateau length before the large noise starts. One new best removes this many generations from the count.
    stagnationBoost: 3, // Noise multiplier at the first plateau, then again after each further stretch of the same length.
    stagnationBoostCap: 64, // Large-noise ceiling. Limb noise at the cap is 0.4, still near the parent.
    sigmaCap: 0.4, // One weight's noise never exceeds this. At the plateau cap every scout stops here, near a trained weight.
    recordWindow: 16, // Generations in the ending average that sets a record.
    recordMargin: 0.33, // That average must clear the previous record by this much.
    bestEverPairs: 2, // Extra antithetic pairs copied from Best ever with the small noise. 0 turns them off. They are not scouts and they do not enter the Adam step.
    bestEverAdoptMargin: 0.5, // During a plateau, a small-noise child replaces the parent when it beats its twin and the parent average by this much. A Best ever copy only has to beat the parent average. Scouts never become the parent.
    stagnationMutateCap: 0.25, // Share of pairs that receive the large noise once a plateau starts. The rest keep the normal noise.
    limbExplorationScale: 0.05, // Noise on shoulder, elbow, hip, knee, and ankle outputs.
    bodyExplorationScale: 0.02, // Noise on hidden weights. Small, so balance can shift without a rewrite.
    learningRate: 0.01, // Adam step size at generation 0. Was 0.025, 2.5x the hidden-weight noise.
    learningRateFloor: 0.003, // Adam step size never drops below this.
    learningRateHalfLifeGenerations: 8000, // Generations for the Adam step size to halve.
    trainableJointIndexes: [1, 2, 4, 5, 6, 7, 8, 10, 11, 12], // Output joints that receive limb noise. Neck and wrists stay put.
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
    frames: 960, // Length of one scored episode.
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
  hint: {
    pairFraction: 0.25, // Share of antithetic pairs whose joint output is forced while a hint button is on. The last pairs, so they are not also the large-noise scouts.
    matchWeight: 1, // Added to fitness when the network's own output already matches the forced sign. A perfect match adds this.
  },
  workers: {
    browserMin: 1, // Fewest browser workers.
    browserMax: 8, // Most browser workers.
    browserReserve: 1, // CPU cores left unused when sizing the browser pool.
    browserHardwareFallback: 4, // Core count assumed when the browser does not report one.
    browserPairMin: 24, // Fewest antithetic pairs in a browser generation. Was 4: too few samples for ~1100 weights.
    headlessPairs: 48, // Antithetic pairs in a headless generation.
    headlessWorkerMin: 2, // Fewest headless workers.
    headlessWorkerMax: 12, // Most headless workers.
    headlessWorkerReserve: 2, // CPU cores left unused when sizing the headless pool.
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
    torso: 0,
    head: 1,
    upperArmLeft: 2,
    forearmLeft: 3,
    handLeft: 4,
    thighLeft: 5,
    shinLeft: 6,
    footLeft: 7,
    upperArmRight: 8,
    forearmRight: 9,
    handRight: 10,
    thighRight: 11,
    shinRight: 12,
    footRight: 13,
    hip: 14, // Tail circle welded under the torso. Not one of the 14 speed inputs.
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
