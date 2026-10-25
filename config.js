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
const headRayMeters = 4; // No head ray looks farther than this.
const episodeSeconds = 6; // One scored individual, unless the head hits first.
const targetCount = 5; // Head, handR, handL, footR, footL.
const targetStride = 7; // use, dx, dy, sin, cos, then dx and dy again on the wide span.
const targetInputCount = targetCount * targetStride;
const pelvisInputCount = (partCount - 1) * 2; // X and Y of every part except the pelvis, in the torso frame.
const footContactInputCount = 2; // Left and right foot. Inputs only, not a penalty.
const contactGroupCount = 7; // Torso, butt, four leg parts, arms. Hands and feet are not groups. Same amounts enter the net.
const sensedBeforeRays =
  jointSenseCount + 4 + 2 + 4 + targetCount * 2 + pelvisInputCount; // Joints, torso/head sin/cos, two heights, four mass values, x/y velocity of the five targets, then the pelvis-frame positions.
const targetInputIndex = sensedBeforeRays + bodyRayCount + headRayCount;
const sensedInputCount =
  targetInputIndex +
  targetInputCount +
  1 +
  footContactInputCount +
  contactGroupCount; // Head contact, the two feet, then one pressure per contact group.
const rewardAxes = 5; // dx, dy, rotation, close x, close y.
const rewardRowCount = targetCount * rewardAxes + contactGroupCount + 1; // Then the upright row.
const rewardTotalNow = false;
const rewardTotalPrevious = false;
const rewardTotalNext = false;
const rewardTotalInputCount =
  Number(rewardTotalNow) +
  Number(rewardTotalPrevious) +
  Number(rewardTotalNext);
// Walk inputs, appended last: sin and cos of a gait clock, the commanded speed with its sign,
// then three terrain numbers for the climb level: a box ahead (0/1), the distance from the feet
// to its near edge, and its height. All 0 when there is no walk and no box, so a net trained to
// stand is unchanged by them.
const walkInputCount = 6;

export const config = {
  physics: {
    hz: physicsHz, // Steps per simulated second.
    timeStep: 1 / physicsHz, // Seconds advanced by one physics step.
    solverSubsteps: 4, // Box2D solver iterations per step. 2 let the angle limits slip.
  },
  cadence: {
    actionEveryNFrames: 6, // Frames an action is held before the net picks a new one. 5 = 12 decisions/s (was 10 = 6/s, slow for balance).
    rewardEveryNFrames: 10, // Frames between score samples. At 60 Hz this is 6 samples a second.
    rewardFrameOffset: 1, // Sample the score this many frames after a new action, not on the action frame.
  },
  net: {
    hidden: [47, 31, 40, 0], // Counts, input toward the outputs. The last layer is the recurrent one. These two widths are what 0.3 and 0.2 built at 156 inputs.
    feedbackUnits: 0, // Nonlinear summary of the previous action. It enters the recurrent layer only.
    walkSkip: true, // Each joint output also reads the three walk inputs directly, so a rhythm is a one-layer problem. Zero-initialised.
    skipTargetInputs: false, // Each joint output also reads the target columns directly, not only through the hidden units. That makes "move this joint toward this command" a one-layer problem. Off for now: it was tried and it hurt skill 1 (0.45 at gen 480 against 0.95 at gen 160 without it), and there is no evidence it helps. See the README.
    jointCount: 14, // Motor joints, and the width of the action vector. Waist, neck, and six joints on each side.
    sensedInputCount, // Observation length before optional reward-total columns.
    targetInputIndex, // First of the target columns. Five targets, seven numbers each: use, dx, dy, error sin, error cos, wide dx, wide dy.
    targetInputCount,
    targetCount,
    headContactInputIndex: targetInputIndex + targetInputCount, // 1 when Box2D says the head is touching. The two foot contacts follow it.
    mutedTargets: [], // Target ids whose five input columns are written as off. The reward still counts them, so this measures what the net loses without them. Used by probe-targets.js.
    partCount, // Dynamic body parts. The tail is extra and is not one of them.
    rewardRowCount, // Score rows shown on screen. They are not copied into the observation.
    rewardTotalNow, // Append the reward total from this decision.
    rewardTotalPrevious, // Append the reward total from the previous decision, one action interval ago.
    rewardTotalNext, // Append the reward total estimated one physics frame ahead.
    walkInputCount, // Gait clock sin, cos, and signed commanded speed. Last in the observation.
    walkInputIndex: sensedInputCount + rewardTotalInputCount,
    inputCount: sensedInputCount + rewardTotalInputCount + walkInputCount, // Full observation length.
    freshInputExplorationScale: 0.04, // Noise on columns after sensedInputCount (reward totals, when on).
    hiddenLeak: 0.75, //0.6, // Fraction of the previous hidden state kept each step.
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
    neckOffsetMeters: neckY - 1.14, // Neck joint above the lower-torso center, in the torso frame. The head arcs around this.
    // Pivots and reach. The arm and the leg start FULLY EXTENDED, so the neutral hand sits
    // exactly at armReachMeters from the shoulder and the neutral foot at legReachMeters from
    // the hip. Any command that pushes a limb further out than that is not a hard command, it
    // is an impossible one, so every commanded position is clamped back onto the reach sphere.
    // These must match the joint positions in ragdoll.js.
    shoulderMeters: 1.48, // Shoulder joint height in the standing blueprint.
    hipMeters: 0.96, // Hip joint height in the standing blueprint.
    armReachMeters: 0.57, // Shoulder to hand centre, arm straight: 0.28 + 0.26 + 0.03.
    legReachMeters: 0.945, // Hip to foot centre, leg straight: 0.44 + 0.42 + 0.085.
    velocityScale: 4, // Metres per second that maps to a full-scale input for part velocity. A diving head hits 2-3 m/s, a swinging hand 4-5.
    waistForwardRadians: (12 * Math.PI) / 180, // Chest flexes this far toward the toes.
    waistBackRadians: (6 * Math.PI) / 180, // Chest extends this far backward. Less than the forward bend.
    hipBackRadians: (22 * Math.PI) / 180, // Thigh extends this far behind standing. Was 40°.
    hipForwardRadians: (128 * Math.PI) / 180, // Thigh flexes this far toward the chest. Was 137°.
    bodyRayMeters: 0.35, // Short rays from the body. 1 means nothing is within this distance of that face.
    pelvisFrameMeters: 1, // A part's position in the torso frame is divided by this before it enters the net.
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
  // span is where that row's position reward reaches its floor, and weight is what it is worth
  // there. Per target, so a hand 15 cm off its mark costs about what a foot 15 cm off costs.
  // ride means the mark is kept in the torso frame: it follows the pelvis while the episode runs,
  // so the row measures posture and not the spot the doll happened to be dropped at.
  // Turns are world level: a ride target's angle does not follow the torso, so it asks for a
  // level head and level hands, which is what the neck and the wrists are for.
  //
  // The x/y below are also the ANATOMICAL NEUTRAL: where that part sits on a standing doll,
  // measured from the pelvis. A command is that neutral plus a random offset inside cmdX/cmdY
  // and cmdAngle. That is the whole difference between a pose holder and a controller: the
  // command has to be able to ask for something the doll is not already doing.
  // cmdArc means "do not sample the position, derive it from the commanded angle". The head
  // capsule pivots headRadius+headAxis away from the neck joint, so its body-frame position
  // can only ever sit on a 14 cm arc. Sampling a box there would ask for the impossible.
  // spanCmd and gainCmd are the numbers for the level that COMMANDS this part: tighter, and heavier.
  // The span and weight above are for the levels with no command. They are different questions,
  // so they are different numbers: raising the head's global weight changed skill 1, where
  // nothing is commanded and the only job is to stand.
  // gainCmd multiplies config.targetGains, which is the manual ramp.
  targets: [
    {
      id: "head",
      part: "head",
      on: true,
      x: 0,
      y: headCenterY,
      angle: 0,
      span: 0.25,
      spanCmd: 0.08,
      weight: 1,
      gainCmd: 3,
      ride: true,
      cmdAngle: 0.5,
      cmdX: 0,
      cmdY: 0,
      cmdArc: true,
    },
    {
      id: "handR",
      part: "handRight",
      on: false,
      x: 0,
      y: 0.91,
      angle: 0,
      span: 0.25,
      spanCmd: 0.15,
      weight: 1,
      gainCmd: 2,
      ride: true,
      cmdAngle: 0.6,
      cmdX: 0.22,
      cmdY: 0.18,
    },
    {
      id: "handL",
      part: "handLeft",
      on: false,
      x: 0,
      y: 0.91,
      angle: 0,
      span: 0.25,
      spanCmd: 0.15,
      weight: 1,
      gainCmd: 2,
      ride: true,
      cmdAngle: 0.6,
      cmdX: 0.22,
      cmdY: 0.18,
    },
    {
      id: "footR",
      part: "footRight",
      on: false,
      x: 0.06,
      y: 0.04,
      angle: 0,
      span: 0.5,
      spanCmd: 0.5,
      weight: 2,
      gainCmd: 1,
      ride: false,
      cmdAngle: 0,
      cmdX: 0,
      cmdY: 0,
    },
    {
      id: "footL",
      part: "footLeft",
      on: false,
      x: 0.06,
      y: 0.04,
      angle: 0,
      span: 0.5,
      spanCmd: 0.5,
      weight: 2,
      gainCmd: 1,
      ride: false,
      cmdAngle: 0,
      cmdX: 0,
      cmdY: 0,
    },
  ],
  // Which group of targets gets a real command at each level of the ladder. Anything not
  // listed here sits on its anatomical neutral, which is what standing already is.
  // 1 stand, 2 head, 3 hands, 4 feet, 5 all five at once.
  // 6 walk: no foot marks, and the head and hand marks slide forward at a commanded speed.
  // 7 mix: each training episode is a stand (1) or a walk (6), so one net keeps both. Eval runs both.
  // 8 climb: a box ahead, the marks slide forward and rise by its height. Trained mixed with walks and stands.
  commands: ["", "none", "head", "hand", "foot", "all", "walk", "mix", "climb"],
  climb: {
    boxHalfWidth: 3, // A 6 m platform. At 2 m a good climber walked off the far end before the episode ended and scored as not on top.
    nearMin: 0.45, // Near edge of the box, ahead of the feet midpoint, in metres.
    nearMax: 0.8,
    minHeight: 0.02, // Training heights are drawn between minHeight and the curriculum cap.
    startCap: 0.05, // Curriculum cap at the start of a climb run.
    maxCap: 0.6, // The cap never goes past this.
    band: 0.04, // Training heights are drawn in [cap - band, cap]: at the frontier, not mostly trivial 2-5 cm boxes.
    capStep: 0.02, // The cap rises this much when recent training climbs succeed often enough.
    capPassRate: 0.25, // Was 0.4: the noisy children sat at 22-27% while the parent climbed 2-3 cm in 10 of 12. Share of climb episodes in the last capWindow generations that must succeed.
    capWindow: 10,
    evalHeights: [0.1, 0.15, 0.2, 0.25, 0.3], // Held-out eval boxes, cycled.
    speedMin: 0.2, // m/s. The marks move a little slower than on flat ground.
    speedMax: 0.35,
    seconds: 8, // Climb episodes are longer than the 6 s default.
    liftZone: 0.6, // Was 0.35: the trailing foot of a wide split stance never entered it. Approach zone before the near edge where a raised foot is paid (reward.climbLift).
    liftRamp: 0.3, // A mark rises over this many metres of x as it passes the near edge.
    onTopSlack: 0.03, // Both feet must be on top to pass. A foot is on top when its centre is at least box top - this, inside the box in x.
    mixClimb: 0.5, // Skill 8 training: share of climb episodes. The rest split between walks and stands.
    mixWalk: 0.25,
  },
  walk: {
    minSpeed: 0.2, // m/s. The marks of head and hands slide at a speed drawn in this range.
    maxSpeed: 0.45,
    delaySeconds: 0.5, // The marks stay put this long, so the doll settles before it has to move.
    periodSeconds: 0.9, // One full gait-clock cycle: a left step and a right step.
    dirs: [1], // Walk directions drawn. The doll faces +x and the hip extends only 22 degrees, so backwards is a different, harder skill.
    footNearMeters: 0.02, // Walk standing check only: a foot this close to the floor counts as down. The reward still uses real touch.
    maxStandSpeed: 1.2, // isStanding's centre-of-mass speed limit while walking. 0.5 when standing.
    passShare: 0.6, // A walk episode passes when the torso covered this share of the commanded distance...
    standShare: 0.6,
    mixWalkShare: 0.5, // Skill 7: share of training episodes that are walks. The rest are stands. // ...and the doll was standing for this share of the last window, and is up at the end.
  },
  // How much of its own row each target puts into the reward. The ramp for turning one group on
  // without disturbing the others: 0 takes it out of the score and out of the network's inputs.
  targetGains: { head: 1.3, handR: 1.3, handL: 1.3, footR: 1.5, footL: 1.5 },

  // CONTACT PENALTIES. Four rows, one per group, each a penalty from 0 down to -weight.
  //
  // Hands and feet are deliberately NOT here: standing on your feet and touching the floor with
  // your hands is allowed. What is not allowed is the body folding onto itself.
  //
  // Each group's amount is the MEAN of its parts. One thigh down costs its share of the row,
  // and lifting any of them moves the number. Every part fully pressed is still -weight.
  // Separation and impulse both rise smoothly (see sim.contactSample): the first impulse is not an instant 1.
  // Revolute spring on the 14 motor joints. This Box2D build has no target angle, so the
  // spring pulls each joint back to the pose it was built in. The speed motor still runs.
  joint: {
    hertz: 2,
    hipHertz: 2, // 0 turns the hip spring off. A passed 0 used to fall through to joint.hertz, so the thigh stayed held down.
    kneeHertz: 2, // Softer than the rest. 20 Hz held the knee straight against the motor, so a step never left the ground.
    dampingRatio: 6.66,
  },
  contact: {
    weight: 2, // Per group, times the group's share. The four leg rows share one weight, so the worst case stays 4 * weight.
    fullMeters: 0.04, // Separation at which a contact counts as fully pressed together.
    impulseFull: 0.79, // Normal impulse of a solid rest, measured on the lying pose in check-contact.js. Amount from impulse is 1 - exp(-impulse / this), so that rest sits near 0.63 and a light touch does not.
    groups: [
      { id: "torso", parts: ["torso", "chest"] },
      { id: "butt", parts: ["hip"] },
      { id: "thighL", parts: ["thighLeft"], share: 0.25 },
      { id: "thighR", parts: ["thighRight"], share: 0.25 },
      { id: "shinL", parts: ["shinLeft"], share: 0.25 },
      { id: "shinR", parts: ["shinRight"], share: 0.25 },
      {
        id: "arms",
        parts: ["upperArmLeft", "upperArmRight", "forearmLeft", "forearmRight"],
      },
    ],
  },
  reward: {
    // Raise this whenever the fitness formula changes. A saved train-state from another
    // version has a bestEverScore on another scale, and the parent never beats it.
    version: 22,
    targetPosition: 1, // Fallback weight for a target that does not carry its own. Feet and body both set one now.
    targetRotation: 1, // Cosine of the angle error. +this when aligned, -this when flipped. Only while that target is on and its gain is above zero.
    footPosition: 2, // Fallback for a foot with no weight of its own. config.targets is where the real numbers live.
    wideWeight: 0.5, // Extra copy of a ride target's position row, on wideSpanMeters. Feet do not get one.
    wideSpanMeters: 1.5, // Span of that copy. The tight span has already flattened by the time a fall reaches this.
    upright: 1.5, // torsoCos + head/standingHead - 1. +this standing, -this lying flat.
    walkSpeed: 3, // Walk level only: +this when the centre of mass moves at the commanded speed. In the upright row.
    climbLift: 1.5, // Climb level only: a foot in the approach zone before the edge earns this times its height over the box top (capped at 1).
    climbFoot: 2, // Was 1, light next to walkSpeed 3. Climb level only: +this per foot on top of the box. In the upright row.
    planted: 1.5, // Added to the upright row: +this with both feet down, -this with one up. Walk level: +this with at least one foot down, -this with both up (no hopping).
    // The score is the target rows, the upright row, and four contact rows. The body marks
    // are planted upright and then frozen, and a penalised part on the floor pays its contact
    // row. Hands and feet are not penalised.
    targetMean: 1, // Global scale. 1 leaves the rows as they are.
    closeWeight: 0.5, // Per axis. The row is this times how many meters that axis closed since the previous reward sample. The first sample is zero.
    startHoldShare: 0.5, // Fraction of a drop from the opening window that is charged. A gain is free.
  },
  curriculum: {
    // These rows drive the *sandbox* start pose and the held-out eval situations (sim.situation).
    // Training does NOT run this ladder: training runs config.skill.level 1-4 through
    // train-headless.js. Do not raise `level` until that skill's held-out eval has passed.
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
  // Skill ladder the training start actually runs. The phase menu above is not this.
  // 1 stand, 2 head, 3 hands, 4 feet, 5 all five. Do not raise `level` until that skill's held-out eval has passed.
  skill: {
    level: 2,
    trainSeconds: 6, // Scored training length when the ramp is off. The held-out eval always uses episode.seconds.
    rampStartSeconds: 1.5, // Duration ramp: the first training episode is this long.
    // 0 turns it off. Asking the search not to fall for 6 s from generation 0 is a balance
    // problem with no steps. It starts short and reaches episode.seconds over rampGenerations.
    // The reward does not change: the same rows, over a shorter episode. The eval ALWAYS runs
    // the full length, or a net that can only stand for 1.5 s would pass.
    rampGenerations: 400, // 0 turns the ramp off.
    stanceMeters: 0.11, // Half the stance width the feet spawn with. Applied to the POSE (options.stance),
    // not the mark. fillStandFeet used to put it on the target dx, and that leaked in as a step
    // command: skill 1 is called STAND, and the right foot was asked to step 11 cm.
    placeStanceMeters: 0.32, // Where each foot already is when it stands. One mark stays here; the other steps away.
    footGroundMeters: 0.04, // Foot-center height once the sole is on the floor. Marks sit here, not at the drop height.
    footSpanMeters: 0.5, // Foot position reward and foot dx/dy use this instead of targetSpanMeters.
    placeMinMeters: 0.12, // Skill 2. Forward shift of the stance midpoint, in meters. Held-out eval uses 0.12–0.16 m.
    placeMaxMeters: 0.16,
    placeScale: 1, // Training shift uses the same 0.12–0.16 m forward range as the held-out eval.
    placeAssist: 0, // Reserved. A joint shove knocked the stand over, so training does not push the foot.
    stepMinMeters: 0.26, // Skills 3 and 4. One foot's step. Long enough that the box is not under the resting sole.
    stepMaxMeters: 0.34,
    stepTurnRadians: 0.25, // Max foot-mark rotation away from flat.
    footReachMeters: 0.06, // A foot is on its mark inside this distance. Also under a third of the average step.
    footReachRadians: 0.3, // And inside this angle error.
    // "Obeyed" tolerances. They have to be SMALLER than the average command, or the bar
    // is met without obeying: a 3 cm average command and an 8 cm tolerance lets a net that
    // never moves pass every time. These are about a third of the error of ignoring the command.
    targetReachMeters: 0.05, // A ride target obeys inside this distance.
    targetReachRadians: 0.15, // And inside this angle error. A third of the average command.
    residualSearch: false, // At skill 2 and up, search the small residual leg net and leave the main one frozen. Off means the main net, the one that reads the target columns, keeps training and a loaded residual stays a frozen leg prior.
    scriptedLegs: false, // The hand written gait in sim.applyScriptedGait overwrites hip, knee, and ankle on its own clock. It cannot obey a foot command, so the command ladder runs without it. Turn it on again for a pure locomotion skill.
    cmdFootMeters: 0.1, // A foot command: half range of the step the foot is asked to take, in meters. Feet are commanded in world axes, they live on the floor.
    oneFoot: true, // Only one foot is asked to step. The other mark stays under that foot.
    swingContact: 0.15, // Below this, a foot is in the air. Its Y row and close Y are zero while the other foot is planted and the torso is up.
    stanceContact: 0.3, // The supporting foot has to be at least this pressed, or both-feet-off still pays Y.
    swingTorsoCos: 0.5, // And the torso has to still be up. A fall does not count as a swing.
    crouchChance: 0.35, // Training foot episodes that start with both knees bent. The held-out eval does not.
    crouchKneeRadians: 0.4, // That bend, at hip and knee. Negative of this is inside the knee limit.
    walkSteps: 3, // Assisted-walk pass: this many steps, then still standing.
    planSteps: 4, // Steps offered in one walk or box episode. The wire format stores this many slots.
    boxHeightMeters: 0.04, // Full height of a floor box. A flat foot has to lift over it.
    boxHalfWidthMeters: 0.04,
    boxAlong: 0.75, // Box center as a fraction of the step, measured from the foot's previous mark.
  },
  search: {
    explorationScale: 0.06, // Target-column noise never drops below this.
    explorationFloor: 0.12, // Starting amplitude of the decaying target noise. The used scale is the max of this decay and explorationScale.
    explorationHalfLifeGenerations: 4000, // Generations for the decaying noise to halve.
    stagnationGenerations: 100, // Plateau length before the large noise starts. One new best removes this many generations from the count.
    stagnationBoost: 2, // Noise multiplier at the first plateau, then again after each further stretch of the same length.
    stagnationBoostCap: 4, // Two boosts, then sigmaCap already clips limb noise.
    sigmaCap: 0.4, // One weight's noise never exceeds this. At the plateau cap every scout stops here, near a trained weight.
    recordWindow: 16, // Generations in the ending average that sets a record.
    recordMargin: 0.2, // That average must clear the previous record by this much.
    bestEverPairs: 0, // Extra pairs copied from Best ever with small noise. 0 turns them off. With this at 0
    // there was still a bug: net.js tell() ranked EVERY child, and the promotion guard only
    // looked at eliteScore, so a child of the anchor could become the parent. The guard now
    // also requires the child to come from the parent.
    bestGenPairs: 1, // Extra pairs copied from last generation's best child, small noise. Re-scored; not the parent.
    elitism: false, // Promote a child that beat the record on one episode. Off: see net.js tell().
    bestEverAdoptMargin: 0, // Was 0.25. 0 turns adoptAnchor off: it replaced the parent on single lucky episodes every generation. // During a plateau, a small-noise child replaces the parent when it beats its twin and the parent average by this much. A Best ever copy only has to beat the parent average. Scouts never become the parent.
    stagnationMutateCap: 0, // Was 0.2. The episode ramp lowers the score on its own, so 'plateau' fired during the ramp. // Share of pairs that receive the large noise once a plateau starts. 0 keeps every pair on the small noise.
    limbExplorationScale: 0.05, // Noise on shoulder, elbow, hip, knee, and ankle outputs.
    walkSkipExplorationScale: 0.15, // Noise on the direct walk-input -> joint weights. Their inputs are 0 off the walk level.
    skipExplorationScale: 0.02, // Noise on the zero-initialised shortcut from the target columns to the joint outputs. Small on purpose: at the full target scale it perturbed every joint's action and the doll could not learn to stand at all.
    bodyExplorationScale: 0.02, // Noise on hidden weights. Small, so balance can shift without a rewrite.
    learningRate: 0.002, // Adam step size at generation 0. A tenth of the hidden-weight noise.
    learningRateFloor: 0.0005, // Adam step size never drops below this.
    learningRateHalfLifeGenerations: 8000, // Generations for the Adam step size to halve.
    revertAfterGenerations: 0, // Was 400. It reverted to a record set on the short ramp episodes. // A plateau this long copies Best ever back onto the parent and clears Adam. 0 leaves the parent where it wandered.
    trainableJointIndexes: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13], // All 14: waist, neck, shoulders, elbows, wrists, hips, knees, ankles. The neck and the wrists used to be frozen, which left the head and both hand rotation rows unreachable.
    adamBeta1: 0.9, // Adam first-moment decay.
    adamBeta2: 0.999, // Adam second-moment decay.
    adamEpsilon: 1e-8, // Added under the Adam denominator.
    hiddenInitScale: 0, // A new parent is all zeros, so tanh is 0 and every joint starts at the standing pose.
    outputInitScale: 0, // Same for the joint outputs. A wide gaussian saturates tanh and locks each joint at a random stop.
    randomHiddenScale: 11, //0.26, // Random-weights button. Gaussian scale of a trained hidden block.
    randomOutputScale: 11.22, // Random-weights button. Gaussian scale of a trained joint-output block.
    // How close to the marker a child must be before it can become the parent. Required only
    // in situations that HAVE a marker (walk, box, ball): it has to have actually arrived.
    // This used to be -1, which is less than every distance, so (gap ?? 9) < -1 was always
    // false and elitism was off in every situation. net.js now skips the guard when the
    // situation has no marker, so the five target levels do not need it, and this value is
    // a real distance again for the ones that do.
    goalCloseEnoughMeters: 0.45, // Elite replace also needs the best child at least this close to the marker.
    eliteMissingGapMeters: 9, // Gap assumed when a score has no marker distance.
    headImpactQuiet: 1.5, // A stand counts as quiet when the peak head-down speed is below this, in m/s.
  },
  episode: {
    seconds: episodeSeconds, // How long one scored individual runs. The episode keeps going after a head touch.
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
    pairs: 48, // Antithetic pairs in one generation, browser and headless.
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
    gateTargets: false, // Also require the head and both hands on their marks before a skill counts as passed. Off, the numbers are still measured and printed every eval.
    nearPassSlack: 0.15, // A rate this close to the pass rate still triggers a full eval.
    // The STAND bar. An episode passes when the doll is standing for THIS fraction of the
    // last endWindowSeconds (3 s = 180 frames), and it is still standing at the end.
    //
    // The old bar was windowStood === windowFrames: standing for all 180 frames, with no miss.
    // One stumble in three seconds failed the episode, so the bar measured "never blinks"
    // rather than "stands". Measured on the gen 474 parent with the old gate: stoodUp 8/20
    // and gate 1/20. Same doll, same net, same seeds: the difference was the bar.
    standWindowShare: 0.9,
    passRate: {
      stand: 0.8,
      head: 0.5, // Standing, and the head inside 8 cm and 0.4 rad of the angle it was told to hold.
      hand: 0.4, // Both hands on their commanded spot as well.
      foot: 0.3, // Both feet too. Harder: the leg has to lift to reach a step it cannot slide to.
      all: 0.25, // All five at once, from the same command every episode. This is the real test.
      feet: 0.6, // Old ladder, kept for the records.
      assist: 0.5,
      boxes: 0.45,
      ball: 0.7,
      drop: 0.65,
      supine: 0.6,
      prone: 0.6,
      walk: 0.6,
      climb: 0.5, // Skill 8: ends with a foot on top of the box, upright, the head never touched anything.
      mix: 1, // Skill 7: both the stand bar and the walk bar (its rate is the worse of the two, as a share of its bar).
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
    hip: 15, // Tail circle welded on the pelvis. Not one of the speed inputs.
  },
};

export function applySearchNumbers(values) {
  if (!values) return;
  const search = config.search;
  const keys = Object.keys(values);
  for (let i = 0; i < keys.length; i++) {
    const key = keys[i];
    if (typeof search[key] !== "number") continue;
    const value = +values[key];
    if (Number.isFinite(value)) search[key] = value;
  }
}

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

export function setSkill(level) {
  const id = level | 0;
  config.skill.level = id < 1 ? 1 : id > 8 ? 8 : id;
}

export function activeTrainingPhase() {
  return (
    config.curriculum.phases.find(
      (item) => item.id === config.curriculum.phase,
    ) || config.curriculum.phases[0]
  );
}
