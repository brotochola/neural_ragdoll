// Every tuning number lives here. The sim, the reward, and the trainer read these names.

export const physicsHz = 60;
export const physicsTimeStep = 1 / physicsHz;
// Box2D splits each frame into this many smaller solves. 4 was stiffer; 2 is the faster setting.
export const solverSubsteps = 2;

// The body holds an action for this many frames, then picks a new one.
export const actionEveryNFrames = 10;
// The score is sampled this often, not on the same frame as the new action.
export const rewardEveryNFrames = 10;
export const rewardFrameOffset = 5;

export const hiddenUnitCount = 20;
export const jointCount = 13;
// Previous joint count of inputs, kept so older saved weights can be padded.
export const legacyInputCount = 79;
export const inputCount = 82;
export const hiddenLeak = 0.25;

export const platformHalfWidth = 20;
export const platformHalfThickness = 0.1;
export const platformPeriodSeconds = 8;

export const touchGapMeters = 0.005;
export const comfortableStanceMeters = 0.29;
export const standingHeadMeters = 1.45;
export const standingTorsoAngleRadians = (15 * Math.PI) / 180;
export const standingStanceMinMeters = 0.05;
export const standingStanceMaxMeters = 0.9;
export const headHeightMeters = 1.68;
export const torsoHeightMeters = 1.26;
export const handClearMeters = 0.15;
export const thighClearMeters = 0.35;
export const thighSpreadMinRadians = (10 * Math.PI) / 180;
export const thighSpreadMaxRadians = (40 * Math.PI) / 180;
export const balanceSpanMeters = 0.4;
export const goalSpanMeters = 2.5;
export const goalWalkSpeed = 0.5;
export const goalCloseEnoughMeters = 0.45;

// Each reward row is multiplied by its weight. Positive rows can reach +weight when the pose is ideal.
export const rewardWeight = {
  torsoAngle: 1,
  headAngle: 0.75,
  upperArmAngle: 0.12,
  forearmAngle: 0.08,
  thighAngle: 0.5,
  shinAngle: 0.25,
  headHeight: 2,
  torsoHeight: 1.8,
  handOnFloor: 0.15,
  handPark: 1.5,
  thighOnFloor: 0.5,
  thighSpread: 10,
  footInAir: 1.5,
  balance: 1.6,
  headImpact: 2.8,
  standing: 5,
  standingFlag: 3,
  stanceComfort: 0.4,
  goalDistanceUp: 0, //2,
  goalDistanceDown: 0, //0.7,
  goalSpeedUp: 0, //1.2,
  goalSpeedDown: 0, //0.4,
};

export const explorationScale = 0.18;
export const explorationFloor = 0.22;
export const explorationHalfLifeGenerations = 6000;
export const limbExplorationScale = 0.05;
export const learningRate = 0.05;
export const learningRateFloor = 0.02;
export const learningRateHalfLifeGenerations = 8000;
// Shoulder, elbow, hip, knee, ankle on both sides. The rest of the net stays put.
export const trainableJointIndexes = [1, 2, 4, 5, 6, 7, 8, 10, 11, 12];

export const episodeFrames = 960;
export const uprightStartChance = 0.55;
export const startTiltJitterRadians = 0.12;
export const startShiftMeters = 0.4;
export const startDropJitterMeters = 0.06;
export const motorGain = 12;
export const motorSpeedLimit = 10;
export const goalNearMeters = 1.2;
export const goalFarExtraMeters = 1;

export const partIndex = {
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
};
