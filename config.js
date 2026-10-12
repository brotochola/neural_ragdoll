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
const sensedBeforeRays = jointSenseCount + 4 + 2 + 4 + targetCount * 2; // Joints, torso/head sin/cos, two heights, four mass values, then x/y velocity of each of the five targets.
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
    skipTargetInputs: false, // Cada salida de articulacion tambien lee las 25 columnas de los targets, directo, no solo por las unidades escondidas. Con esto "mover esta articulacion a esta orden" es un problema de una capa. OFF de momento: se probo y empeoro skill 1 (0.45 a gen 480 contra 0.95 a gen 160 sin el), y no hay evidencia de que ayude. Ver README.
    jointCount: 14, // Motor joints, and the width of the action vector. Waist, neck, and six joints on each side.
    sensedInputCount, // Observation length before optional reward-total columns.
    targetInputIndex, // First of the target columns. Five targets, five numbers each: use, dx, dy, error sin, error cos.
    targetInputCount,
    targetCount,
    headContactInputIndex: sensedInputCount - 1, // 1 when Box2D says the head is touching.
    mutedTargets: [], // Target ids whose five input columns are written as off. The reward still counts them, so this measures what the net loses without them. Used by probe-targets.js.
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
    neckOffsetMeters: neckY - 1.14, // Neck joint above the lower-torso center, in the torso frame. The head arcs around this.
    // Pivotes and reach. The arm and the leg start FULLY EXTENDED, so the neutral hand sits
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
  // spanCmd y gainCmd son los numeros del nivel que MANDA a esta parte: mas apretado y mas peso.
  // Los de arriba (span, weight) son los de los niveles donde no hay orden. Son preguntas
  // distintas, asi que son numeros distintos: subir el peso global de la cabeza le cambiaba el
  // paisaje a skill 1, donde no hay nada que mandar y solo importa pararse.
  // gainCmd multiplica a config.targetGains, que es la rampa manual.
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
  commands: ["", "none", "head", "hand", "foot", "all"],
  // How much of its own row each target puts into the reward. The ramp for turning one group on
  // without disturbing the others: 0 takes it out of the score and out of the network's inputs.
  targetGains: { head: 1, handR: 1, handL: 1, footR: 1, footL: 1 },
  reward: {
    // Súbele este número cada vez que cambie la fórmula del fitness. Un train-state guardado
    // con otra versión tiene un bestEverScore de otra escala, y el padre no lo supera nunca.
    version: 6,
    targetPosition: 1, // Fallback weight for a target that does not carry its own. Feet and body both set one now.
    targetRotation: 1, // Cosine of the angle error. +this when aligned, -this when flipped. Only while that target is on and its gain is above zero.
    footPosition: 2, // Fallback for a foot with no weight of its own. config.targets is where the real numbers live.
    // EL PREMIO SON SOLO LAS DIEZ FILAS DE LOS TARGETS. Ni upright, ni headTouch, ni alive,
    // ni fail, ni motion, ni step. Todo lo de abajo esta en cero y fuera de la formula.
    // "Pararse" no necesita ser un termino aparte: cae solo por los targets. Las marcas se
    // plantan UNA vez, con el torso parado, y quedan congeladas en el mundo. Si el torso se
    // cae, la cabeza baja a 0.12 m contra una marca a 1.58 m, y eso YA es el premio.
    // Medido con node check-collapse.js: parado 7.92 contra tumbado 4.74 de un techo de 12.
    // Ver README.
    targetMean: 1, // Escala global del fitness. 1 deja las diez filas tal cual.
    startHoldShare: 0.5, // Cuanto se le cobra de lo que se perdio entre el arranque y el final.
    // SOLO la perdida: si el doll MEJORA durante el episodio no se le cobra nada. Antes la mezcla
    // era (1-share)*vida + share*min(arranque, vida), y como min siempre es <= vida, eso le
    // cobraba justo a quien mejoraba: dar un paso con el pie subia el promedio del episodio y el
    // fitness se quedaba a medio camino de no haberlo dado. Ver sim.js fitnessValue.
    headTouch: 0, // RETIRADO del fitness. Queda solo como dato en el readout.
    motion: 0, // RETIRADO.
    upright: 0, // RETIRADO.
    alive: 0, // RETIRADO.
    aliveShare: 0, // RETIRADO.
    fail: 0, // RETIRADO.
    step: 0, // RETIRADO.
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
  // 1 stand, 2 feet on a short step, 3 assisted steps, 4 those steps over a low box.
  // Do not raise `level` until that skill's held-out eval has passed.
  skill: {
    level: 2,
    trainSeconds: 6, // Scored training length when the ramp is off. The held-out eval always uses episode.seconds.
    rampStartSeconds: 1.5, // La rampa de DURACION del entrenamiento: el primer episodio dura esto.
    // 0 la apaga. Pedir "no caerse" durante 6 s en la generacion 0 es un control de equilibrio
    // sin peldaños. Arranca corto y llega a episode.seconds en rampGenerations. El PREMIO no
    // cambia: son las mismas diez filas sobre un episodio más corto. El eval SIEMPRE corre entero,
    // o se estaría aprobando una red que solo sabe pararse 1.5 s.
    rampGenerations: 400, // 0 la apaga.
    stanceMeters: 0.11, // Half the stance width the feet spawn with. Se aplica a la POSE (options.stance),
    // no a la marca. Antes fillStandFeet lo ponía en el dx del target y eso se colaba como ORDEN
    // de paso: en el skill 1, que es PARADO, al pie derecho se le pedía un paso de 11 cm.
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
    // Tolerancias de "obedecio". Tienen que ser MAS CHICAS que la orden promedio, o la barra
    // se cumple sin obedecer: con una orden de 3 cm de promedio y una tolerancia de 8 cm, un net
    // que se queda quieto pasa siempre. Estas son cerca de un tercio del error de ignorar.
    targetReachMeters: 0.05, // A ride target obeys inside this distance.
    targetReachRadians: 0.15, // And inside this angle error. Un tercio del comando promedio.
    residualSearch: false, // At skill 2 and up, search the small residual leg net and leave the main one frozen. Off means the main net, the one that reads the target columns, keeps training and a loaded residual stays a frozen leg prior.
    scriptedLegs: false, // The hand written gait in sim.blendLegs overwrites hip/knee/ankle on its own clock. It cannot obey a foot command, so the command ladder runs without it. Turn it on again for a pure locomotion skill.
    cmdFootMeters: 0.22, // A foot command: half range of the step the foot is asked to take, in meters. Feet are commanded in world axes, they live on the floor.
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
    stagnationGenerations: 1000, // Plateau length before the large noise starts. One new best removes this many generations from the count.
    stagnationBoost: 3, // Noise multiplier at the first plateau, then again after each further stretch of the same length.
    stagnationBoostCap: 9, // Two boosts, then sigmaCap already clips limb noise.
    sigmaCap: 0.4, // One weight's noise never exceeds this. At the plateau cap every scout stops here, near a trained weight.
    recordWindow: 16, // Generations in the ending average that sets a record.
    recordMargin: 0.2, // That average must clear the previous record by this much.
    bestEverPairs: 0, // Pares extra copiados de Best ever con ruido chico. 0 los apaga. Ojo: con esto en 0
    // igual habia un bug. En net.js tell(), order se ordenaba sobre TODOS los hijos y el guard de
    // promocion solo miraba eliteScore, sin preguntar de quien venia: un hijo del ancla podia
    // llegar a ser padre. Ahora el guard exige ademas que sea hijo del padre.
    bestEverAdoptMargin: 0.25, // During a plateau, a small-noise child replaces the parent when it beats its twin and the parent average by this much. A Best ever copy only has to beat the parent average. Scouts never become the parent.
    stagnationMutateCap: 0, // Share of pairs that receive the large noise once a plateau starts. 0 keeps every pair on the small noise.
    limbExplorationScale: 0.05, // Noise on shoulder, elbow, hip, knee, and ankle outputs.
    skipExplorationScale: 0.02, // Noise on the zero-initialised shortcut from the target columns to the joint outputs. Small on purpose: at the full target scale it perturbed every joint's action and the doll could not learn to stand at all.
    bodyExplorationScale: 0.02, // Noise on hidden weights. Small, so balance can shift without a rewrite.
    learningRate: 0.002, // Adam step size at generation 0. A tenth of the hidden-weight noise.
    learningRateFloor: 0.0005, // Adam step size never drops below this.
    learningRateHalfLifeGenerations: 8000, // Generations for the Adam step size to halve.
    revertAfterGenerations: 400, // A plateau this long copies Best ever back onto the parent and clears Adam. 0 leaves the parent where it wandered.
    trainableJointIndexes: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13], // All 14: waist, neck, shoulders, elbows, wrists, hips, knees, ankles. The neck and the wrists used to be frozen, which left the head and both hand rotation rows unreachable.
    adamBeta1: 0.9, // Adam first-moment decay.
    adamBeta2: 0.999, // Adam second-moment decay.
    adamEpsilon: 1e-8, // Added under the Adam denominator.
    hiddenInitScale: 0.12, // Fresh hidden weights are a gaussian times this.
    outputInitScale: 0.04, // Fresh joint-output weights are a gaussian times this.
    randomHiddenScale: 0.26, // Random-weights button. Gaussian scale of a trained hidden block.
    randomOutputScale: 0.22, // Random-weights button. Gaussian scale of a trained joint-output block.
    // Distancia a la meta que un hijo tiene que tener para poder ser el padre. SOLO se exige en
    // las situaciones que TIENEN meta (caminar, caja, pelota): ahi tiene que haber llegado de verdad.
    // Este numero estaba en -1, y -1 es menor que toda distancia, asi que el guard era
    // (gap ?? 9) < -1 = falso SIEMPRE y la elitancia estaba apagada en todas las situaciones.
    // net.js ahora saltea el guard cuando la situacion no tiene meta, asi que los cinco niveles
    // de targets no lo necesitan; y este valor vuelve a ser una distancia de verdad para las que si.
    goalCloseEnoughMeters: 0.45, // Elite replace also needs the best child at least this close to the marker.
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
    gateTargets: false, // Also require the head and both hands on their marks before a skill counts as passed. Off, the numbers are still measured and printed every eval.
    nearPassSlack: 0.15, // A rate this close to the pass rate still triggers a full eval.
    // LA BARRA DE "PARADO". Un episodio pasa si el doll esta de pie durante ESTA fraccion de los
    // ultimos endWindowSeconds (3 s = 180 frames), y ademas mantiene la racha de pie al final.
    //
    // Antes la barra era windowStood === windowFrames: de pie en los 180 frames, sin uno menos. Un
    // solo tropiezo en tres segundos echaba el episodio, y con eso la barra no media "se para" sino
    // "se queda clavado sin parpadear". Medido sobre el padre de la gen 474 con el gate viejo:
    // stoodUp 8/20 y gate 1/20. El mismo doll, la misma red, las mismas semillas: la diferencia
    // entera era la barra, no la red.
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

export function setSkill(level) {
  const id = level | 0;
  config.skill.level = id < 1 ? 1 : id > 5 ? 5 : id;
}

export function activeTrainingPhase() {
  return (
    config.curriculum.phases.find(
      (item) => item.id === config.curriculum.phase,
    ) || config.curriculum.phases[0]
  );
}
