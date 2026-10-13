import {
  CreateMouseJoint,
  b2MouseJoint_SetTarget,
  b2DestroyJoint,
  b2Body_GetPosition,
  b2Body_GetRotation,
  b2Rot_GetAngle,
  b2Vec2,
  b2World_Draw,
  b2DebugDraw,
} from "https://cdn.jsdelivr.net/npm/phaser-box2d@1.1.0/dist/PhaserBox2D.min.js";
import * as Box2D from "https://cdn.jsdelivr.net/npm/phaser-box2d@1.1.0/dist/PhaserBox2D.min.js";
import { config, applySearchNumbers } from "./config.js";
import { scoreDecision, isStanding } from "./src/reward.js";
import { ElmanNetwork, EvolutionSearch } from "./src/net.js";
import {
  clamp,
  makeSeededRandom,
  randomNormalSample,
  rotate,
  quantizeWeights,
} from "./src/utils.js";
import {
  createSim,
  observationLabels,
  jointNames,
  thrownKinds,
} from "./src/sim.js";
import { RagDoll2D } from "./src/ragdoll.js";
import {
  openStore,
  loadStored,
  putSession,
  syncSnapshots,
  storeReady,
} from "./src/session-store.js";

const QUAD_X = [-1, 1, 1, -1];
const QUAD_Y = [-1, -1, 1, 1];
const GRAPH_ALPHA = [0.06, 0.11, 0.18, 0.3, 0.46, 0.7];

export class NeuralRagDollWorld {
  constructor() {
    this.start();
    window.neuralRagDollWorld = this;
  }

  async start() {
    this.$ = (id) => document.getElementById(id);
    this.sim = createSim(Box2D);
    this.sim.setRandom(Date.now() & 0xffff || 1);
    this.random = makeSeededRandom(7);
    this.randomNormal = () => randomNormalSample(this.random);
    this.state = await this.openSession();
    this.paused = true;
    this.phaseStartGen = this.state.gen || 0;
    this.backupDir = null;
    this.backedBestScore = this.state.bestEverScore ?? -1e9;
    this.visit = { log: [], parentName: "", loadedLabel: "", trained: false };
    if (this.discardedNetwork) {
      this.saveSession();
      this.clearSnapshots();
      this.toast(
        "Saved network doesn't match this layout. Starting from scratch.",
      );
      this.logEvent("stale network discarded");
    }
    this.point = new b2Vec2(0, 0);
    this.cursor = { x: 0, y: 0 };
    this.pick = { x: 0, y: 0 };
    this.rotated = { x: 0, y: 0 };
    this.boxPoly = [0, 0, 0, 0, 0, 0, 0, 0];

    for (let i = 0; i < thrownKinds.length; i++) {
      const option = document.createElement("option");
      option.value = String(i);
      option.textContent = thrownKinds[i].name;
      this.$("objectKind").appendChild(option);
    }
    this.fillSavedList();
    this.targets = config.targets.map((item) => ({
      on: item.on,
      x: item.x,
      y: item.y,
      angle: item.angle,
      placed: item.on,
    }));
    this.trainMode = false;
    this.trainSaved = false;
    this.jitter = 0.05;
    this.randomStart = true;
    this.trainMarks = config.targets.map((item) => ({
      on: item.part === "head",
      x: item.x,
      y: item.y,
      angle: item.angle || 0,
      placed: item.part === "head",
    }));
    this.targetColors = [0xffd166, 0xff7a18, 0xb14dff, 0x3dff6a, 0x2ec8ff];
    this.targetDrag = null;
    this.fillTargetToggles();
    this.restoreUi();
    this.restoreSearch();
    this.fillSearchKnobs();

    this.doll = new RagDoll2D(Box2D, config);
    this.world = this.doll;
    this.episode = undefined;
    this.hold = 0;
    this.objects = [];
    this.drag = null;
    this.aim = null;
    this.view = { cx: 0, scale: 100, groundY: 0, width: 1, height: 1 };
    this.$("pause").onclick = () => {
      this.paused = !this.paused;
      this.paintPause();
      this.saveUi();
      this.saveSession();
      this.postOrchestrator({ type: "pause", paused: this.paused });
      this.logEvent(this.paused ? "training paused" : "training started");
    };
    this.$("skillLevel").onchange = () => this.saveUi();
    this.$("scoutOnce").onclick = () => {
      if (!this.orchestrator) return;
      this.scoutPending = true;
      this.postOrchestrator({ type: "scout" });
      if (this.state) this.paintTraining();
    };
    this.$("playMode").onchange = () => {
      this.trainMode = this.$("playMode").value === "train";
      this.syncTrainBoxes();
      this.paintTargetToggles();
      this.syncEpisodeTargets();
      if (this.episode) this.episode.passive = !this.trainMode && this.$("passive").checked;
      if (this.trainMode) this.postTraining();
      this.saveUi();
    };
    this.$("trainJitter").oninput = () => {
      this.jitter = +this.$("trainJitter").value || 0;
      this.paintJitter();
      this.saveUi();
      if (this.trainSaved) this.postTraining();
    };
    this.$("randomStart").onchange = () => {
      this.randomStart = this.$("randomStart").checked;
      this.saveUi();
      if (this.trainSaved) this.postTraining();
    };
    window.addEventListener("pagehide", () => {
      this.saveSession();
      this.saveUi();
    });
    document.addEventListener("visibilitychange", () => {
      if (document.visibilityState === "hidden") this.saveSession();
    });
    this.paintPause();
    this.$("restart").onclick = () => {
      this.freshSearch();
      this.logEvent("restart");
    };
    this.$("reset").onclick = () => this.resetEpisode();
    this.$("saveFolder").onclick = async () => {
      if (!window.showDirectoryPicker) {
        this.$("error").textContent = "This browser has no folder picker.";
        return;
      }
      try {
        this.backupDir = await window.showDirectoryPicker({
          mode: "readwrite",
        });
        this.$("saveFolder").textContent = "Folder set";
        this.$("error").textContent = "";
        this.logEvent("save-folder chosen");
      } catch (error) {
        if (error && error.name === "AbortError") return;
        this.$("error").textContent = "Could not use that folder.";
      }
    };
    this.$("loadNetwork").onclick = () => {
      const input = this.$("loadNetworkFile");
      input.value = "";
      input.click();
    };
    this.$("loadNetworkFile").onchange = async () => {
      const file =
        this.$("loadNetworkFile").files && this.$("loadNetworkFile").files[0];
      if (!file) return;
      let data;
      try {
        data = JSON.parse(await file.text());
      } catch (error) {
        this.$("error").textContent = "Could not read that network file.";
        this.logEvent("JSON load failed");
        return;
      }
      const revived = this.networkFromJson(data);
      if (!revived) {
        this.freshSearch();
        this.paused = true;
        this.paintPause();
        this.toast(
          "That file doesn't match this network. Starting from scratch.",
        );
        this.logEvent("JSON load failed");
        return;
      }
      const stem = file.name.replace(/\.json$/i, "");
      let hash = "";
      try {
        hash = await this.thetaHash8(revived.theta);
      } catch (error) {
        hash = "";
      }
      this.visit.loadedLabel =
        hash && !stem.endsWith(hash) ? stem + " " + hash : stem;
      this.visit.parentName = "";
      this.visit.trained = false;
      if (hash) revived.networkName = this.fileStamp(new Date()) + "_" + hash;
      if (data && data.role === "best-ever" && !revived.bestEverName)
        revived.bestEverName = stem;
      this.state = revived;
      this.backedBestScore = this.state.bestEverScore ?? -1e9;
      this.paused = true;
      this.paintPause();
      this.$("error").textContent = "";
      this.saveSession();
      this.syncSnapshotStore();
      this.fillSavedList();
      this.$("savedNet").value = "";
      this.$("watchParent").checked = true;
      this.$("watchGeneration").checked = false;
      this.$("watchEver").checked = false;
      this.resetEpisode();
      this.pushSearch();
      this.saveUi();
      this.logEvent("JSON load succeeded");
    };
    this.$("randomWeights").onclick = () => {
      const raw = window.prompt(
        "This throws away the saved network and starts over.\nRandom scale (0–1 is normal, 10 is extreme):",
        "1",
      );
      if (raw == null) return;
      const scale = +raw;
      if (!Number.isFinite(scale) || scale < 0) return;
      this.state = new EvolutionSearch(this.randomNormal);
      const outputBase = ElmanNetwork.layout.outputBase;
      for (let i = 0; i < outputBase; i++)
        this.state.theta[i] = this.randomNormal() * scale;
      for (let i = outputBase; i < this.state.theta.length; i++)
        this.state.theta[i] = this.randomNormal() * scale;
      this.state.bestGenTheta = this.state.theta.slice();
      this.state.bestEverTheta = this.state.theta.slice();
      this.state.snapshots = [];
      this.backedBestScore = this.state.bestEverScore ?? -1e9;
      this.visit.parentName = "";
      this.visit.loadedLabel = "";
      this.visit.trained = false;
      this.fillSavedList();
      this.$("savedNet").value = "";
      this.$("watchParent").checked = true;
      this.$("watchGeneration").checked = false;
      this.$("watchEver").checked = false;
      this.clearSnapshots();
      this.saveSession();
      this.pushSearch();
      this.resetEpisode();
      this.paintTraining();
      this.logEvent("random weights scale " + scale);
    };
    this.$("log").onclick = () => {
      this.paintLog();
      this.toggleDialog(this.$("logDialog"));
      if (this.$("logDialog").open)
        this.$("logBody").scrollTop = this.$("logBody").scrollHeight;
    };
    this.$("logClose").onclick = () => this.$("logDialog").close();
    this.$("graphBtn").onclick = () => {
      const dialog = this.$("graphDialog");
      if (dialog.open) dialog.close();
      else {
        dialog.show();
        requestAnimationFrame(() => {
          this.drawNetworkGraph();
          this.paintNetList();
        });
      }
    };
    this.$("graphClose").onclick = () => this.$("graphDialog").close();
    this.choosingWatch = false;
    this.watchLive = (event) => {
      if (this.choosingWatch || !event.target.checked) return;
      this.choosingWatch = true;
      this.$("savedNet").value = "";
      this.choosingWatch = false;
      this.resetEpisode();
      this.saveUi();
      this.paintTraining();
    };
    this.$("watchParent").onchange = this.watchLive;
    this.$("watchGeneration").onchange = this.watchLive;
    this.$("watchEver").onchange = this.watchLive;
    this.$("savedNet").onchange = () => {
      if (this.choosingWatch) return;
      this.choosingWatch = true;
      const saved = this.$("savedNet").value;
      this.$("watchParent").checked = !saved;
      this.$("watchGeneration").checked = false;
      this.$("watchEver").checked = false;
      this.choosingWatch = false;
      this.resetEpisode();
      this.saveUi();
      this.paintTraining();
    };
    this.$("speed").oninput = () => {
      this.$("speedValue").textContent = this.$("speed").value + " m/s";
      this.saveUi();
    };
    for (const id of [
      "snapshotEvery",
      "startPose",
      "viewRays",
      "viewColliders",
      "passive",
      "objectKind",
      "showTrainPose",
    ]) {
      this.$(id).addEventListener("change", () => this.saveUi());
    }

    this.app = new PIXI.Application();
    this.host = this.$("stage");
    await this.app.init({
      resizeTo: this.host,
      background: 0x12162a,
      antialias: true,
      resolution: window.devicePixelRatio || 1,
      autoDensity: true,
    });
    this.host.appendChild(this.app.canvas);
    this.graphics = new PIXI.Graphics();
    this.throws = new PIXI.Graphics();
    this.colliders = new PIXI.Graphics();
    this.dollView = new PIXI.Container();
    this.dollView.sortableChildren = true;
    this.label = new PIXI.Text({
      text: "",
      style: {
        fill: 0xdfe4ff,
        fontSize: 13,
        fontFamily: "system-ui,sans-serif",
        lineHeight: 18,
      },
    });
    this.label.position.set(12, 10);
    this.app.stage.addChild(
      this.graphics,
      this.dollView,
      this.throws,
      this.colliders,
      this.label,
    );
    this.doll.attachView(this.dollView);
    this.prepareColliderDraw();
    this.resetEpisode();
    for (const id of ["trainHead", "trainFeet", "trainHands"]) {
      this.$(id).onchange = () => {
        this.applyTrainGroup(id);
        this.saveUi();
      };
    }
    this.trainLoop();

    this.canvas = this.app.canvas;
    this.canvas.style.touchAction = "none";
    this.accumulator = 0;
    this.frameSkip = 0;
    this.canvas.addEventListener("pointerdown", (event) => {
      const screen = this.screenPoint(event);
      const scale = this.view.scale || 100;
      const groundY = this.view.groundY || this.app.screen.height * 0.78;
      const point = this.worldPoint(screen.x, screen.y, scale, groundY);
      this.canvas.setPointerCapture(event.pointerId);
      const targetHit = this.pickTarget(point);
      if (targetHit) {
        this.targetDrag = targetHit;
        return;
      }
      let picked = null;
      let best = 1e9;
      const parts = this.world.parts;
      for (let i = 0; i < parts.length; i++) {
        const part = parts[i];
        const dx = part.x - point.x;
        const dy = part.y - point.y;
        const distance = dx * dx + dy * dy;
        if (distance < best && distance < (part.hx + 0.2) ** 2) {
          best = distance;
          picked = part;
        }
      }
      if (picked) {
        this.point.x = point.x;
        this.point.y = point.y;
        const joint = CreateMouseJoint({
          worldId: this.world.worldId,
          bodyIdA: this.world.groundId,
          bodyIdB: picked.id,
          target: this.point,
          hertz: 5,
          dampingRatio: 0.7,
          maxForce: 1500,
        });
        this.drag = {
          id: joint.jointId,
          x: screen.x,
          y: screen.y,
          part: picked,
        };
      } else
        this.aim = { x0: screen.x, y0: screen.y, x: screen.x, y: screen.y };
    });
    this.canvas.addEventListener("pointermove", (event) => {
      const screen = this.screenPoint(event);
      if (this.targetDrag) {
        const scaleNow = this.view.scale || 100;
        const groundNow = this.view.groundY || this.app.screen.height * 0.78;
        const point = this.worldPoint(screen.x, screen.y, scaleNow, groundNow);
        const target = this.markList()[this.targetDrag.index];
        const spec = config.targets[this.targetDrag.index];
        if (this.targetDrag.mode === "turn")
          target.angle = Math.atan2(point.y - target.y, point.x - target.x);
        else {
          // Mundo puro: la marca se queda donde la dejes. No hay offset local que el proximo
          // frame convierta otra vez, y por eso no "salta" cuando mueves el doll.
          target.x = point.x;
          target.y = point.y;
          target.placed = true;
        }
        void spec;
        return;
      }
      if (this.drag) {
        this.drag.x = screen.x;
        this.drag.y = screen.y;
      }
      if (this.aim) {
        this.aim.x = screen.x;
        this.aim.y = screen.y;
      }
    });
    this.canvas.addEventListener("pointerup", () => this.release());
    this.canvas.addEventListener("pointercancel", () => this.release());
    this.$("fire").onclick = () => {
      const side = Math.random() < 0.5 ? -1 : 1;
      const speed = +this.$("speed").value;
      const kind = thrownKinds[+this.$("objectKind").value];
      const id = this.sim.spawnThrown(
        this.world,
        kind,
        this.episode.measure.centerX + side * 3.5,
        1.2,
        -side * speed,
        1.5,
      );
      this.objects.push({ id, kind });
    };

    this.app.ticker.add((ticker) => this.onFrame(ticker));
  }

  onFrame(ticker) {
    const now = performance.now();
    if (this.frameStamp) {
      const gap = now - this.frameStamp;
      if (gap > (this.frameGapMs || 0)) this.frameGapMs = gap;
    }
    this.frameStamp = now;
    this.episode.measure.recordRays = this.$("viewRays").checked;
    this.accumulator += Math.min(ticker.deltaMS, 50);
    let stepSum = 0;
    let stepCount = 0;
    while (this.accumulator >= 1000 / config.physics.hz) {
      this.accumulator -= 1000 / config.physics.hz;
      if (this.hold > 0) {
        this.hold--;
        if (!this.hold) this.resetEpisode();
      } else {
        const stepStart = performance.now();
        this.episode.passive = !this.trainMode && this.$("passive").checked;
        const ended = this.episode.advanceOneFrame();
        stepSum += performance.now() - stepStart;
        stepCount++;
        if (ended) this.hold = 60;
      }
    }
    if (stepCount) this.stepMs = stepSum / stepCount;
    const width = this.app.screen.width;
    const height = this.app.screen.height;
    const scale = Math.min(height / 2.8, 150);
    const groundY = height * 0.78;
    this.view.width = width;
    this.view.height = height;
    this.view.scale = scale;
    this.view.groundY = groundY;
    this.view.cx +=
      ((this.episode.measure.centerX || 0) - this.view.cx) *
      Math.min(1, ticker.deltaMS / 250);
    if (this.drag) {
      const point = this.worldPoint(this.drag.x, this.drag.y, scale, groundY);
      this.point.x = point.x;
      this.point.y = point.y;
      b2MouseJoint_SetTarget(this.drag.id, this.point);
    }
    const renderStart = performance.now();
    this.drawGround();
    this.doll.update(this.view);
    this.throws.clear();
    this.drawThrown();
    this.drawRays();
    this.drawTargets();
    this.drawTrainPreview();
    this.drawColliders();
    this.drawChart();
    this.renderMs = performance.now() - renderStart;
    if (++this.frameSkip % 8 === 0) {
      this.frameGapShow = this.frameGapMs;
      this.frameGapMs = 0;
      const scored = scoreDecision(this.episode.measure);
      this.paintScore(scored);
      if (this.$("graphDialog").open) {
        this.drawNetworkGraph();
        this.paintNetList();
      }
      const pose = isStanding(this.episode.measure)
        ? "standing"
        : this.episode.measure.handOn
          ? "on hands"
          : this.episode.measure.bodyOn
            ? "on the floor"
            : "in the air";
      const hit = this.episode.headHit ? " · head hit" : "";
      this.paintTraining();
      this.label.text =
        "Head " +
        (this.episode.measure.headHeight || 0).toFixed(2) +
        " m · torso " +
        (
          (Math.abs(
            Math.atan2(
              this.episode.measure.torsoSin || 0,
              this.episode.measure.torsoCos ?? 1,
            ),
          ) *
            180) /
          Math.PI
        ).toFixed(0) +
        "° · " +
        pose +
        hit;
    }
  }

  toX(x) {
    const view = this.view;
    return (x - view.cx) * view.scale + view.width * 0.5;
  }

  toY(y) {
    return this.view.groundY - y * this.view.scale;
  }

  drawGround() {
    const width = this.view.width;
    const height = this.view.height;
    const scale = this.view.scale;
    const graphics = this.graphics;
    graphics.clear().rect(0, this.toY(0), width, height).fill(0x232a47);
    const left = this.view.cx - width / scale / 2;
    const right = this.view.cx + width / scale / 2;
    for (let mark = Math.floor(left) - 1; mark <= right + 1; mark++) {
      graphics.rect(this.toX(mark) - 1, this.toY(0), 2, 8).fill(0x3a4570);
    }
  }

  drawThrown() {
    const objects = this.objects;
    const throws = this.throws;
    const corners = this.boxPoly;
    const rot = this.rotated;
    const scale = this.view.scale;
    for (let i = 0; i < objects.length; i++) {
      const thrown = objects[i];
      const position = b2Body_GetPosition(thrown.id);
      const x = position.x;
      const y = position.y;
      if (thrown.kind.box) {
        const angle = b2Rot_GetAngle(b2Body_GetRotation(thrown.id));
        const cosine = Math.cos(angle);
        const sine = Math.sin(angle);
        const half = thrown.kind.half;
        for (let k = 0; k < 4; k++) {
          rotate(QUAD_X[k] * half, QUAD_Y[k] * half, cosine, sine, rot);
          corners[k * 2] = this.toX(x + rot.x);
          corners[k * 2 + 1] = this.toY(y + rot.y);
        }
        // Pixi no copia el array: si reusamos boxPoly, todas las cajas quedan en la ultima.
        throws.poly(corners.slice()).fill(thrown.kind.color);
      } else
        throws
          .circle(this.toX(x), this.toY(y), thrown.kind.radius * scale)
          .fill(thrown.kind.color);
    }
  }

  drawRays() {
    if (!this.$("viewRays").checked) return;
    const doll = this.doll;
    const count = doll.rayCount;
    const x0 = doll.rayX0;
    const y0 = doll.rayY0;
    const x1 = doll.rayX1;
    const y1 = doll.rayY1;
    const gap = doll.rayGap;
    const hit = doll.rayHit;
    const throws = this.throws;
    for (let i = 0; i < count; i++) {
      const touched = hit[i] !== 0;
      const distance = gap[i];
      const color = !touched
        ? 0x8b93b0
        : distance < 0.02
          ? 0xff4d5a
          : distance < 0.35
            ? 0xffb020
            : 0x3ecf8e;
      throws
        .moveTo(this.toX(x0[i]), this.toY(y0[i]))
        .lineTo(this.toX(x1[i]), this.toY(y1[i]))
        .stroke({ width: 1.5, color, alpha: touched ? 0.95 : 0.35 });
      if (touched)
        throws.circle(this.toX(x1[i]), this.toY(y1[i]), 3).fill(color);
    }
  }

  trainLoop() {
    const worker = new Worker("./src/orchestrator.js", { type: "module" });
    this.orchestrator = worker;
    this.epoch = 0;
    worker.onmessage = (event) => this.onOrchestrator(event.data);
    worker.onerror = () => {
      this.$("error").textContent = "Could not start training";
    };
    this.pushSearch();
  }

  postOrchestrator(message, transfer) {
    if (!this.orchestrator) return;
    if (transfer) this.orchestrator.postMessage(message, transfer);
    else this.orchestrator.postMessage(message);
  }

  searchValues() {
    const out = {};
    const search = config.search;
    const keys = Object.keys(search);
    for (let i = 0; i < keys.length; i++) {
      const key = keys[i];
      if (typeof search[key] === "number") out[key] = search[key];
    }
    return out;
  }

  restoreSearch() {
    let saved;
    try {
      saved = JSON.parse(localStorage.getItem("ragdoll-search") || "null");
    } catch (error) {
      return;
    }
    applySearchNumbers(saved);
  }

  fillSearchKnobs() {
    const tips = {
      explorationScale: "Floor of the noise on the weights that read the target columns. The decaying noise never goes below this, so the search keeps trying the five marks even late in a run.",
      explorationFloor: "Starting size of the target-column noise. It decays with the generation count, and the noise actually used is the larger of that decay and explorationScale.",
      explorationHalfLifeGenerations: "How many generations it takes for the decaying target noise to fall to half. A smaller number makes the search get quieter faster.",
      stagnationGenerations: "Generations without a new Best ever before the large noise can start. The Plateau row says Stagnation at this count. One new Best ever subtracts this many generations from the count.",
      stagnationBoost: "Multiplier on the noise of the scout pairs at the first plateau, and again after each further stretch of stagnationGenerations. Scout once uses this same multiplier.",
      stagnationBoostCap: "Largest multiplier the plateau is allowed to reach. sigmaCap can stop a weight sooner than this.",
      sigmaCap: "No single weight is noised by more than this, scout or not. At the plateau cap the scouts sit near a trained weight instead of jumping past it.",
      recordWindow: "How many recent generations are averaged to decide Best ever. The average is of the parent’s children, not of the single best child. A window that is not full yet does not set a record.",
      recordMargin: "The record average must beat the previous Best ever by at least this much. A smaller margin accepts tinier improvements and resets the plateau sooner.",
      bestEverPairs: "Extra pairs copied from Best ever and noised with the small noise. 0 turns them off. They are added on top of the parent’s pairs.",
      bestGenPairs: "Extra pairs copied from last generation’s best child, with the small noise. They are scored again. They do not become the parent. 0 turns them off.",
      bestEverAdoptMargin: "During Stagnation, a small-noise child replaces the parent when it beats its twin and the parent average by this much. A Best ever copy only has to beat the parent average. A scout never becomes the parent.",
      stagnationMutateCap: "Share of the parent’s pairs that get the large noise once Stagnation starts, and when you press Scout once. 0.2 is one pair in five. 0 keeps every pair on the small noise, except Scout once still scouts one pair.",
      limbExplorationScale: "Noise on the shoulder, elbow, hip, knee, and ankle outputs. The waist, neck, and wrists use this too while they are in the trainable list.",
      skipExplorationScale: "Noise on the shortcut from the target columns straight to the joint outputs. Those weights start at zero. A large value here shakes every joint and the doll stops being able to stand.",
      bodyExplorationScale: "Noise on the hidden weights. Kept small so balance can shift without rewriting the whole net.",
      learningRate: "Adam’s step size at generation 0. Each generation moves the parent by about this much in the direction the children ranked. It then decays.",
      learningRateFloor: "Adam’s step size never falls below this, no matter how many generations have passed.",
      learningRateHalfLifeGenerations: "How many generations it takes for Adam’s step size to fall to half, down to learningRateFloor.",
      revertAfterGenerations: "A plateau this long copies Best ever back onto the parent and clears Adam’s memory. 0 leaves the parent where it wandered.",
      adamBeta1: "How fast Adam forgets the direction of recent steps. Closer to 1 remembers longer. 0.9 is the usual first-moment decay.",
      adamBeta2: "How fast Adam forgets the size of recent steps. Closer to 1 remembers longer. 0.999 is the usual second-moment decay.",
      adamEpsilon: "A tiny number added under Adam’s division so a weight that has never moved does not divide by zero.",
      hiddenInitScale: "A brand new parent draws its hidden weights from a gaussian of this width. Restart from scratch uses it. A run that is already going does not.",
      outputInitScale: "A brand new parent draws its joint-output weights from a gaussian of this width. Restart from scratch uses it. A run that is already going does not.",
      randomHiddenScale: "The Random weights button replaces the hidden block with a gaussian of this width, the scale of a trained network rather than a fresh one.",
      randomOutputScale: "The Random weights button replaces the joint-output block with a gaussian of this width.",
      goalCloseEnoughMeters: "In a situation that has a marker, a child must be at least this close to it before it can become the parent. The five target levels have no marker, so they ignore this.",
      eliteMissingGapMeters: "Distance assumed when a score has no marker. It only matters for the marker check, which the target levels skip.",
      headImpactQuiet: "A stand counts as quiet when the fastest the head was moving downward stays below this, in meters per second.",
    };
    const root = this.$("searchKnobList");
    const search = config.search;
    const keys = Object.keys(search);
    for (let i = 0; i < keys.length; i++) {
      const key = keys[i];
      if (typeof search[key] !== "number") continue;
      const label = document.createElement("label");
      const tip = tips[key] || key;
      const name = document.createElement("span");
      name.textContent = key;
      name.title = tip;
      const input = document.createElement("input");
      input.type = "number";
      input.step = "any";
      input.value = String(search[key]);
      input.title = tip;
      input.oninput = () => {
        const value = +input.value;
        if (!Number.isFinite(value)) return;
        applySearchNumbers({ [key]: value });
        try {
          localStorage.setItem("ragdoll-search", JSON.stringify(this.searchValues()));
        } catch (error) {
          /* a quota error still leaves this generation's number in memory */
        }
        this.postOrchestrator({ type: "search", values: this.searchValues() });
        if (this.state) this.paintTraining();
      };
      label.append(name, input);
      root.appendChild(label);
    }
  }

  markList() {
    return this.trainMode ? this.trainMarks : this.targets;
  }

  paintJitter() {
    const cm = Math.round((this.jitter || 0) * 100);
    this.$("jitterVal").textContent = cm + " cm";
  }

  postTraining() {
    this.trainSaved = true;
    const spec = this.trainSpec();
    const state = this.state;
    if (state && this.trainKey !== undefined && this.trainKey !== this.trainFingerprint(spec)) {
      state.eliteScore = -1e9;
      state.bestEverScore = -1e9;
      state.bestSmooth = null;
      state.flatGens = 0;
      if (Array.isArray(state.hist)) state.hist.length = 0;
    }
    this.trainKey = this.trainFingerprint(spec);
    this.postOrchestrator({ type: "targets", ...spec });
  }

  trainFingerprint(spec) {
    const parts = [];
    const marks = spec.targets || [];
    for (let i = 0; i < marks.length; i++) {
      const mark = marks[i] || {};
      parts.push(
        (mark.on ? 1 : 0) +
          "," +
          (mark.x || 0) +
          "," +
          (mark.y || 0) +
          "," +
          (mark.angle || 0),
      );
    }
    parts.push(+spec.jitter || 0);
    parts.push(spec.randomStart !== false ? 1 : 0);
    return parts.join("|");
  }

  trainSpec() {
    return {
      targets: this.trainMarks.map((item) => ({
        on: !!item.on,
        x: item.x || 0,
        y: item.y || 0,
        angle: item.angle || 0,
      })),
      jitter: this.jitter,
      randomStart: this.randomStart,
    };
  }

  postTargets() {
    if (this.trainMode) this.postTraining();
  }

  searchPacket() {
    const state = this.state;
    const theta = state.theta.slice();
    const bestGenTheta = state.bestGenTheta.slice();
    const bestEverTheta = state.bestEverTheta.slice();
    const m = state.m.slice();
    const v = state.v.slice();
    return {
      packet: {
        theta,
        bestGenTheta,
        bestEverTheta,
        m,
        v,
        k: state.k || 0,
        gen: state.gen || 0,
        flatGens: state.flatGens || 0,
        bestEverScore: state.bestEverScore,
        bestGenScore: state.bestGenScore,
        bestSmooth: state.bestSmooth,
        eliteScore: state.eliteScore,
        parentMoves: state.parentMoves || 0,
        parentMoveNote: state.parentMoveNote || "",
        hist: state.hist || [],
      },
      transfer: [
        theta.buffer,
        bestGenTheta.buffer,
        bestEverTheta.buffer,
        m.buffer,
        v.buffer,
      ],
    };
  }

  pushSearch() {
    if (!this.orchestrator || !this.state) return;
    this.epoch = (this.epoch || 0) + 1;
    const sent = this.searchPacket();
    const message = {
      type: "replace",
      epoch: this.epoch,
      paused: this.paused,
      phase: config.curriculum.phase,
      state: sent.packet,
    };
    if (this.trainSaved) Object.assign(message, this.trainSpec());
    message.search = this.searchValues();
    this.scoutPending = false;
    this.postOrchestrator(message, sent.transfer);
  }

  onOrchestrator(data) {
    if (!data) return;
    if (data.type === "error") {
      this.$("error").textContent = data.message || "Training stopped";
      this.paused = true;
      this.paintPause();
      return;
    }
    if (data.type !== "generation" || data.epoch !== this.epoch) return;
    const state = this.state;
    if (!state || data.theta.length !== state.theta.length) return;
    state.theta.set(data.theta);
    state.bestGenTheta.set(data.bestGenTheta);
    state.bestEverTheta.set(data.bestEverTheta);
    if (data.m && data.m.length === state.m.length) state.m.set(data.m);
    if (data.v && data.v.length === state.v.length) state.v.set(data.v);
    state.gen = data.gen || 0;
    state.k = data.k || 0;
    state.flatGens = data.flatGens || 0;
    if (data.scouted) this.scoutPending = false;
    state.mutationBoost = data.mutationBoost || 1;
    state.mutationFraction = data.mutationFraction || 0;
    state.bestEverScore = data.bestEverScore;
    state.bestGenScore = data.bestGenScore;
    state.bestSmooth = data.bestSmooth;
    state.eliteScore = data.eliteScore;
    const previousMoves = state.parentMoves || 0;
    state.parentMoves = data.parentMoves || 0;
    state.parentMoveNote = typeof data.parentMoveNote === "string" ? data.parentMoveNote : "";
    if (state.parentMoves > previousMoves && state.parentMoveNote)
      this.logEvent(state.parentMoveNote);
    if (data.row) {
      state.hist.push(data.row);
      if (state.hist.length > 400) state.hist.shift();
    }
    this.sendMs = data.sendMs;
    this.tellMs = data.tellMs;
    this.workerTiming = data.timings || [];
    if (data.preview) this.trainPreview = this.sim.previewStart(this.world, data.preview);
    if (data.anchorNote) this.logEvent(data.anchorNote);
    this.visit.trained = true;
    const finished = state.hist[state.hist.length - 1];
    if (finished) {
      const mut = this.mutationStatus(", ");
      const began = Number.isFinite(finished.start)
        ? finished.start.toFixed(2) + " → "
        : "";
      this.logEvent(
        "generation finished (gen " +
          state.gen +
          ", " +
          began +
          finished.mean.toFixed(2) +
          ", gen best " +
          finished.best.toFixed(2) +
          mut +
          ")",
      );
    }
    this.rememberBest();
    if (state.gen % config.session.everyGenerations === 0) this.saveSession();
  }

  savedSnapshot() {
    const generation = this.$("savedNet").value;
    if (!generation) return null;
    return (
      (this.state.snapshots || []).find(
        (item) => String(item.gen) === generation,
      ) || null
    );
  }

  watchedWeights() {
    const saved = this.savedSnapshot();
    if (saved) return saved.weights;
    if (this.$("watchEver").checked)
      return this.state.bestEverTheta || this.state.theta;
    if (this.$("watchGeneration").checked)
      return this.state.bestGenTheta || this.state.theta;
    return this.state.theta;
  }

  formatMs(value) {
    return Number.isFinite(value) ? value.toFixed(2) + " ms" : "–";
  }

  formatRange(low, high, digits) {
    if (!Number.isFinite(low)) return "–";
    const a = low.toFixed(digits);
    const b = high.toFixed(digits);
    return a === b ? a : a + "–" + b;
  }

  setPerf(index, text) {
    const node = this.perfValues[index];
    if (node.textContent !== text) node.textContent = text;
  }

  setStatus(index, text) {
    const node = this.statusValues[index];
    if (node.textContent !== text) node.textContent = text;
  }

  setStatusTip(index, tip) {
    const row = this.statusTips[index];
    if (row.title !== tip) row.title = tip;
  }

  fillStats(root, specs) {
    const values = [];
    const tips = [];
    root.textContent = "";
    for (let i = 0; i < specs.length; i++) {
      const spec = specs[i];
      const row = document.createElement("div");
      row.className = spec[3] ? "stat " + spec[3] : "stat";
      row.title = spec[1];
      const label = document.createElement("span");
      if (spec[2]) {
        const mark = document.createElement("i");
        mark.className = "swatch " + spec[2];
        label.appendChild(mark);
      }
      label.appendChild(document.createTextNode(spec[0]));
      const value = document.createElement("b");
      row.append(label, value);
      root.appendChild(row);
      values.push(value);
      tips.push(row);
    }
    return { values, tips };
  }

  ensureStatus() {
    if (this.statusValues) return;
    const every = config.search.stagnationGenerations;
    const windowGens = config.search.recordWindow;
    const startSec = config.episode.startWindowSeconds;
    const endSec = config.episode.endWindowSeconds;
    const hold = config.reward.startHoldShare;
    const perf = this.fillStats(this.$("perf"), [
      [
        "Step",
        "Milliseconds for one physics step of the doll on screen. This is the main thread, not a worker.",
      ],
      [
        "Render",
        "Milliseconds to draw the frame on the main thread, after the physics step. Does not include rewriting this panel.",
      ],
      [
        "Frame gap",
        "Milliseconds between frames on the main thread. This is the wait you feel. Step and Render are only the work inside the frame.",
      ],
    ]);
    this.perfValues = perf.values;
    const status = this.fillStats(this.$("status"), [
      [
        "Generation",
        "How many generations have finished. Each generation is a batch of children of the parent, scored on one episode each.",
      ],
      [
        "Start → end",
        "Average reward of the parent's children in the first " +
          startSec +
          " s, then in the last " +
          endSec +
          " s. Copies of Best ever are not in this average. This is the reward total, not the blue line.",
      ],
      [
        "Avg",
        "Mean fitness of the parent's children, averaged over the last " +
          windowGens +
          " generations. The blue line. Fitness is the reward over the last " +
          endSec +
          " s, minus " +
          hold +
          " of any drop from the opening window. A gain is free. The total adds each target axis, how far that axis closed, the upright row, and the contact rows.",
        "avg",
      ],
      [
        "Generation best",
        "Highest fitness among the children, including the light copies of Best ever. The green line. Network → Best this generation plays that child. The next generation replaces it. This is not Best ever.",
        "best",
      ],
      [
        "Best ever",
        "That same " +
          windowGens +
          "-generation average, the last time it beat the previous record by " +
          config.search.recordMargin +
          ". Network → Best ever plays the parent from that generation. A single lucky child does not set this.",
      ],
      [
        "Elite",
        "The fitness a child of the parent has to beat before it replaces the parent. It only goes up. Each generation is a different command, so one lucky episode can set a bar later children never clear. Adam still steps the parent when this stays put.",
      ],
      [
        "Plateau",
        "Generations since the " +
          windowGens +
          "-generation average fitness last reached a new high. The row says Stagnation once the count reaches " +
          every +
          ". " +
          (config.search.stagnationMutateCap > 0
            ? "At " +
              every +
              ", mutation becomes x" +
              config.search.stagnationBoost +
              " for " +
              Math.round(config.search.stagnationMutateCap * 100) +
              "% of pairs, and that factor multiplies again every " +
              every +
              " generations. "
            : "Scout noise is off. ") +
          "One new Best ever removes " +
          every +
          " from this count. After " +
          config.search.revertAfterGenerations +
          " generations the parent is put back on Best ever and Adam's memory is cleared.",
      ],
      [
        "Mutation",
        config.search.stagnationMutateCap > 0
          ? "How far the scout children are pushed from the parent, and what share of the pairs get that push. The rest keep the small noise. x1 means nobody is scouting. The factor follows the plateau: at " +
            every +
            " generations it becomes x" +
            config.search.stagnationBoost +
            ". Scout noise never passes " +
            config.search.sigmaCap +
            ". The weights that read the target columns and the head-contact column always get the larger noise, so the five targets are the first thing the search explores. When a Best ever record exists, the extra pairs are light copies of that network. A copy becomes the parent when it beats the parent average. A scout never does."
          : "Scout noise is off, so every pair keeps the small noise. x1 means nobody is scouting. When a Best ever record exists, the extra pairs are light copies of that network. A copy becomes the parent when it beats the parent average.",
      ],
      [
        "Send",
        "Milliseconds to copy the weight buffers and hand them to the workers. The workers have not started yet.",
        null,
        "split",
      ],
      [
        "Handoff",
        "Milliseconds from the handoff until the scores come back, minus the time the worker spent simulating. Waiting and copying, not physics.",
      ],
      [
        "Tell",
        "Milliseconds to fold the scores into the parent, on the training worker, after the sim workers finish.",
      ],
      [
        "Save",
        "Milliseconds to hand the session to IndexedDB. The weights are copied as binary, not as text.",
      ],
      [
        "Ragdolls per worker",
        "How many ragdolls each worker ran in the last generation. A pair is two ragdolls. They share a starting pose and try opposite small changes.",
      ],
      [
        "Seconds per generation",
        "How long each worker took to finish every ragdoll it was given, one after another. The workers run at the same time, so a generation lasts about this long.",
      ],
      [
        "Milliseconds per frame",
        "How long one physics frame takes while a ragdoll is moving. A frame is one small update of the body. The picture on screen aims for a frame every 16 milliseconds. The workers run faster than that and do not wait.",
      ],
      // Al FINAL a proposito: las filas de arriba se llenan por indice, y este dato es nuevo.
      [
        "Parent moves",
        "How many times a child has replaced the parent. A plateau can also hand the parent to a small-noise child, and that counts here. If it stays at 0, the population is exploring around a parent that has not been replaced: the average can still move from Adam's step. A target level has no goal, so nothing gates the promotion there; a situation that has a goal also needs the child within " +
          config.search.goalCloseEnoughMeters +
          " m of it.",
      ],
      [
        "Why",
        "The training worker's note for the best child of the parent this generation. A note with 'vs elite' means that child did not clear the bar.",
        null,
        "note",
      ],
    ]);
    this.statusValues = status.values;
    this.statusTips = status.tips;
  }

  paintTraining() {
    this.ensureStatus();
    const last = this.state.hist[this.state.hist.length - 1];
    const ever = this.state.bestEverScore;
    const everText =
      Number.isFinite(ever) && ever > -1e8 ? ever.toFixed(2) : "–";
    const every = config.search.stagnationGenerations;
    const flat = this.state.flatGens || 0;
    const steps = every > 0 && flat >= every ? Math.floor(flat / every) : 0;
    const pct =
      steps > 0 && config.search.stagnationMutateCap > 0
        ? Math.round(Math.min(1, config.search.stagnationMutateCap) * 100)
        : 0;
    const boost =
      pct > 0
        ? Math.min(
            config.search.stagnationBoostCap,
            Math.pow(config.search.stagnationBoost, steps),
          )
        : 1;
    const anchor = this.state.anchorPairCount
      ? this.state.anchorPairCount()
      : 0;
    const anchorText = anchor > 0 ? " · best ever ×" + anchor : "";
    const ending =
      last && Number.isFinite(last.end) ? last.end : last ? last.mean : null;
    const arrow =
      !last || !Number.isFinite(ending)
        ? "–"
        : Number.isFinite(last.start)
          ? last.start.toFixed(2) + " → " + ending.toFixed(2)
          : ending.toFixed(2);
    let handoffLow = Infinity;
    let handoffHigh = -Infinity;
    let handoffCount = 0;
    const timing = this.workerTiming || [];
    let ragLow = Infinity;
    let ragHigh = -Infinity;
    let secLow = Infinity;
    let secHigh = -Infinity;
    let frameLow = Infinity;
    let frameHigh = -Infinity;
    let ragCount = 0;
    let ragTotal = 0;
    for (let i = 0; i < timing.length; i++) {
      const row = timing[i];
      const handoff = row && row.handoff;
      if (handoff >= 0) {
        if (handoff < handoffLow) handoffLow = handoff;
        if (handoff > handoffHigh) handoffHigh = handoff;
        handoffCount++;
      }
      if (!row || !(row.ms >= 0) || !(row.count > 0)) continue;
      ragCount++;
      ragTotal += row.count;
      if (row.count < ragLow) ragLow = row.count;
      if (row.count > ragHigh) ragHigh = row.count;
      const seconds = row.ms / 1000;
      if (seconds < secLow) secLow = seconds;
      if (seconds > secHigh) secHigh = seconds;
      const frame = row.steps > 0 ? row.ms / row.steps : 0;
      if (frame < frameLow) frameLow = frame;
      if (frame > frameHigh) frameHigh = frame;
    }
    const handoffText = !handoffCount
      ? "–"
      : this.formatRange(handoffLow, handoffHigh, 1) + " ms";
    const pending = "–";
    const elite = this.state.eliteScore;
    const eliteText =
      Number.isFinite(elite) && elite > -1e8 ? elite.toFixed(2) : "–";
    const windowGens = config.search.recordWindow;
    let avgText = "–";
    if (this.state.hist.length) {
      const begin = Math.max(0, this.state.hist.length - windowGens);
      let sum = 0;
      let n = 0;
      for (let i = begin; i < this.state.hist.length; i++) {
        if (!Number.isFinite(this.state.hist[i].mean)) continue;
        sum += this.state.hist[i].mean;
        n++;
      }
      if (n) avgText = (sum / n).toFixed(2);
    }
    this.setPerf(0, this.formatMs(this.stepMs));
    this.setPerf(1, this.formatMs(this.renderMs));
    this.setPerf(2, this.formatMs(this.frameGapShow));
    this.setStatus(0, String(this.state.gen || 0));
    this.setStatus(1, arrow);
    this.setStatus(2, avgText);
    this.setStatus(3, last ? last.best.toFixed(2) : "–");
    this.setStatus(4, everText);
    this.setStatus(5, eliteText);
    const stagnant = every > 0 && flat >= every;
    this.setStatus(
      6,
      stagnant ? "Stagnation · " + flat : flat + " / " + (every > 0 ? every : "–"),
    );
    if (this.statusTips[6]) this.statusTips[6].classList.toggle("stagnant", stagnant);
    this.setStatus(
      7,
      (this.scoutPending ? "scout next · " : "") +
        "x" +
        boost +
        (pct > 0 ? " · " + pct + "%" : "") +
        anchorText,
    );
    this.setStatus(8, this.formatMs(this.sendMs));
    this.setStatus(9, handoffText);
    this.setStatus(10, this.formatMs(this.tellMs));
    this.setStatus(11, this.formatMs(this.saveMs));
    this.setStatus(
      12,
      ragCount ? this.formatRange(ragLow, ragHigh, 0) : pending,
    );
    this.setStatus(
      13,
      ragCount ? this.formatRange(secLow, secHigh, 1) : pending,
    );
    this.setStatus(
      14,
      ragCount ? this.formatRange(frameLow, frameHigh, 2) : pending,
    );
    this.setStatus(15, String(this.state.parentMoves || 0));
    this.setStatus(16, this.state.parentMoveNote || "–");
    if (ragCount)
      this.setStatusTip(
        12,
        "How many ragdolls each worker ran in the last generation. There were " +
          ragTotal +
          " ragdolls in total, split across " +
          ragCount +
          " workers. A pair is two ragdolls. They share a starting pose and try opposite small changes. More pairs means more ragdolls here.",
      );
  }
  seatIfNew(target, spec) {
    if (!target.on || target.placed || !this.episode) return;
    const part = config.partIndex[spec.part];
    const measure = this.episode.measure;
    target.x = measure.partX[part];
    target.y = measure.partY[part];
    target.angle = Math.atan2(measure.partSin[part], measure.partCos[part]);
    target.placed = true;
  }

  applyTrainGroup(id) {
    const on = this.$(id).checked;
    const group =
      id === "trainHead" ? "head" : id === "trainFeet" ? "foot" : "hand";
    const list = this.markList();
    config.targets.forEach((spec, index) => {
      if (!spec.id.startsWith(group)) return;
      const target = list[index];
      target.on = on;
      this.seatIfNew(target, spec);
    });
    this.paintTargetToggles();
    this.postTargets();
    this.paintTraining();
  }

  syncTrainBoxes() {
    const list = this.markList();
    const on = (name) => {
      const index = config.targets.findIndex((spec) => spec.id === name);
      return !!(list[index] && list[index].on);
    };
    this.$("trainHead").checked = on("head");
    this.$("trainFeet").checked = on("footL") && on("footR");
    this.$("trainHands").checked = on("handL") && on("handR");
    this.$("passive").disabled = this.trainMode;
  }

  networkTitle() {
    const saved = this.savedSnapshot();
    if (saved) return saved.name || "gen " + saved.gen;
    if (this.$("watchEver").checked) {
      const gen = Number.isFinite(this.state.bestEverGen)
        ? this.state.bestEverGen
        : this.state.gen;
      return (
        "Best ever, generation " +
        gen +
        (this.state.bestEverName ? ", " + this.state.bestEverName : "")
      );
    }
    if (this.$("watchGeneration").checked) {
      return (
        "Best this generation, generation " +
        this.state.gen +
        (this.state.bestGenName ? ", " + this.state.bestGenName : "")
      );
    }
    if (this.visit.parentName) return this.visit.parentName;
    if (this.visit.loadedLabel && this.paused && !this.visit.trained)
      return "Loaded " + this.visit.loadedLabel;
    return "Parent, generation " + this.state.gen;
  }

  logEvent(line) {
    const now = new Date();
    const pad = (n) => String(n).padStart(2, "0");
    const log = this.visit.log;
    log.push(
      pad(now.getHours()) +
        ":" +
        pad(now.getMinutes()) +
        ":" +
        pad(now.getSeconds()) +
        "  " +
        line,
    );
    if (log.length > 80) log.shift();
    const dialog = this.$("logDialog");
    if (!dialog.open) return;
    const body = this.$("logBody");
    body.textContent = log.join("\n");
    body.scrollTop = body.scrollHeight;
  }

  paintLog() {
    this.$("logBody").textContent = this.visit.log.length
      ? this.visit.log.join("\n")
      : "Nothing yet.";
  }

  toggleDialog(dialog) {
    if (dialog.open) dialog.close();
    else dialog.show();
  }

  fillTargetToggles() {
    const row = this.$("targetRow");
    config.targets.forEach((spec, index) => {
      const label = document.createElement("label");
      label.className = "ctrl target-toggle";
      label.title =
        spec.ride
          ? "Command for " +
            spec.id +
            ". Drag it to tell the doll where you want that part. The mark stays in world coordinates: it goes where you put it and it does NOT follow the torso, so you can see the doll miss it. Its tip sets the angle you want in world terms. In training the mark is planted once and frozen the same way, which is why a fall costs you the head and hand rows."
          : "Command for " +
            spec.id +
            ". Drag it to tell the doll where you want that foot. This one lives on the floor, in world coordinates.";
      const input = document.createElement("input");
      input.type = "checkbox";
      input.checked = this.markList()[index].on;
      input.onchange = () => {
        const target = this.markList()[index];
        target.on = input.checked;
        this.seatIfNew(target, spec);
        this.syncTrainBoxes();
        this.postTargets();
        this.saveUi();
      };
      label.append(input, document.createTextNode(" " + spec.id));
      row.appendChild(label);
    });
  }

  paintTargetToggles() {
    const inputs = this.$("targetRow").querySelectorAll("input");
    const list = this.markList();
    for (let i = 0; i < inputs.length && i < list.length; i++)
      inputs[i].checked = !!list[i].on;
  }

  pickTarget(point) {
    let best = null;
    let bestDist = Infinity;
    const stick = 0.34;
    const list = this.markList();
    for (let i = 0; i < list.length; i++) {
      const target = list[i];
      if (!target.on) continue;
      const tipX = target.x + Math.cos(target.angle) * stick;
      const tipY = target.y + Math.sin(target.angle) * stick;
      const tipDist = Math.hypot(point.x - tipX, point.y - tipY);
      if (tipDist < 0.12 && tipDist < bestDist) {
        best = { index: i, mode: "turn" };
        bestDist = tipDist;
      }
      const dist = Math.hypot(point.x - target.x, point.y - target.y);
      if (dist < 0.16 && dist < bestDist) {
        best = { index: i, mode: "move" };
        bestDist = dist;
      }
    }
    return best;
  }

  drawTargets() {
    this.ensureGhosts();
    const stick = 0.34;
    const parts = this.doll.parts;
    const list = this.markList();
    for (let i = 0; i < list.length; i++) {
      const target = list[i];
      const sprite = this.sandboxMarks && this.sandboxMarks[i];
      if (!target.on) {
        if (sprite) sprite.visible = false;
        continue;
      }
      if (sprite) {
        const part = parts[config.partIndex[config.targets[i].part]];
        this.placeGhost(sprite, target.x, target.y, target.angle || 0, part);
        sprite.alpha = 0.5;
      }
      const color = this.targetColors[i];
      const x = this.toX(target.x);
      const y = this.toY(target.y);
      const tipX = this.toX(target.x + Math.cos(target.angle) * stick);
      const tipY = this.toY(target.y + Math.sin(target.angle) * stick);
      this.throws.moveTo(x, y).lineTo(tipX, tipY).stroke({ width: 2, color, alpha: 0.7 });
      this.throws.circle(tipX, tipY, 5).fill(color);
    }
  }

  ensureGhosts() {
    if (this.poseGhosts || !this.doll.partSprites) return;
    this.poseLayer = new PIXI.Container();
    this.poseLayer.zIndex = 0;
    this.dollView.addChildAt(this.poseLayer, 0);
    this.poseGhosts = [];
    const sprites = this.doll.partSprites;
    const count = this.doll.parts.length;
    for (let i = 0; i < count; i++) {
      const src = sprites[i];
      if (!src) {
        this.poseGhosts.push(null);
        continue;
      }
      const ghost = new PIXI.Sprite(src.texture);
      ghost.anchor.set(0.5);
      ghost.alpha = 0.22;
      ghost.tint = src.tint;
      ghost.texW = src.texW;
      ghost.texH = src.texH;
      ghost.visible = false;
      this.poseLayer.addChild(ghost);
      this.poseGhosts.push(ghost);
    }
    this.markLayer = new PIXI.Container();
    this.app.stage.addChildAt(this.markLayer, this.app.stage.getChildIndex(this.throws));
    this.sandboxMarks = [];
    this.previewMarks = [];
    for (let i = 0; i < config.targets.length; i++) {
      const src = sprites[config.partIndex[config.targets[i].part]];
      const sandbox = new PIXI.Sprite(src.texture);
      sandbox.anchor.set(0.5);
      sandbox.alpha = 0.5;
      sandbox.tint = src.tint;
      sandbox.texW = src.texW;
      sandbox.texH = src.texH;
      sandbox.visible = false;
      this.markLayer.addChild(sandbox);
      this.sandboxMarks.push(sandbox);
      const preview = new PIXI.Sprite(src.texture);
      preview.anchor.set(0.5);
      preview.alpha = 0.5;
      preview.tint = src.tint;
      preview.texW = src.texW;
      preview.texH = src.texH;
      preview.visible = false;
      this.markLayer.addChild(preview);
      this.previewMarks.push(preview);
    }
  }

  placeGhost(sprite, x, y, angle, part) {
    const scale = this.view.scale || 1;
    sprite.position.set(this.toX(x), this.toY(y));
    sprite.rotation = -angle;
    this.doll.sizeSprite(sprite, part.hx * 2, part.hy * 2, scale);
  }

  drawTrainPreview() {
    this.ensureGhosts();
    if (!this.poseGhosts) return;
    const show = this.$("showTrainPose").checked && this.trainPreview;
    const preview = this.trainPreview;
    const parts = this.doll.parts;
    for (let i = 0; i < this.poseGhosts.length; i++) {
      const ghost = this.poseGhosts[i];
      if (!ghost) continue;
      if (!show) {
        ghost.visible = false;
        continue;
      }
      const partAngle = Array.isArray(preview.angle)
        ? preview.angle[i]
        : preview.angle;
      this.placeGhost(ghost, preview.x[i], preview.y[i], partAngle, parts[i]);
      ghost.alpha = 0.22;
    }
    for (let i = 0; i < this.previewMarks.length; i++) {
      const sprite = this.previewMarks[i];
      const mark = show && preview.marks[i];
      if (!mark || !mark.on) {
        sprite.visible = false;
        continue;
      }
      this.placeGhost(sprite, mark.x, mark.y, mark.angle, parts[config.partIndex[mark.part]]);
      sprite.alpha = 0.5;
    }
  }

  prepareColliderDraw() {
    const debug = new b2DebugDraw();
    const skip = () => {};
    for (const key of Object.keys(debug)) {
      if (typeof debug[key] === "object" && debug[key] === null)
        debug[key] = skip;
    }
    debug.drawShapes = true;
    debug.drawJoints = false;
    debug.useDrawingBounds = false;
    const stroke = { width: 1.5, color: 0x7dffb3, alpha: 0.95 };
    debug.DrawSolidCapsule = (center1, center2, radius) => {
      const scale = this.view.scale || 1;
      const x1 = this.toX(center1.x);
      const y1 = this.toY(center1.y);
      const x2 = this.toX(center2.x);
      const y2 = this.toY(center2.y);
      const pixels = radius * scale;
      const dx = x2 - x1;
      const dy = y2 - y1;
      const length = Math.hypot(dx, dy) || 1;
      const px = (-dy / length) * pixels;
      const py = (dx / length) * pixels;
      this.colliders.moveTo(x1 + px, y1 + py).lineTo(x2 + px, y2 + py);
      this.colliders.moveTo(x1 - px, y1 - py).lineTo(x2 - px, y2 - py);
      this.colliders.circle(x1, y1, pixels);
      this.colliders.circle(x2, y2, pixels);
      this.colliders.stroke(stroke);
    };
    debug.DrawSolidCircle = (transform, radius) => {
      const scale = this.view.scale || 1;
      const point = transform.p || transform;
      this.colliders
        .circle(this.toX(point.x), this.toY(point.y), radius * scale)
        .stroke(stroke);
    };
    debug.DrawSolidPolygon = (transform, vertices, count) => {
      if (!transform || !transform.p || !transform.q || count < 3) return;
      const cosine = transform.q.c;
      const sine = transform.q.s;
      const originX = transform.p.x;
      const originY = transform.p.y;
      const graphics = this.colliders;
      for (let i = 0; i < count; i++) {
        const vertex = vertices[i];
        const x = this.toX(originX + cosine * vertex.x - sine * vertex.y);
        const y = this.toY(originY + sine * vertex.x + cosine * vertex.y);
        if (i === 0) graphics.moveTo(x, y);
        else graphics.lineTo(x, y);
      }
      graphics.closePath();
      graphics.stroke(stroke);
    };
    this.boxDebug = debug;
  }

  drawColliders() {
    this.colliders.clear();
    if (!this.$("viewColliders").checked) return;
    try {
      b2World_Draw(this.world.worldId, this.boxDebug);
    } catch (error) {
      this.$("viewColliders").checked = false;
      this.colliders.clear();
      this.$("error").textContent = error.message;
    }
  }

  syncEpisodeTargets() {
    if (!this.episode || !this.episode.options) return;
    this.episode.options.targets = this.markList();
  }

  episodeOptions() {
    const pose = this.$("startPose").value;
    const shared = {
      frames: 1e9,
      targets: this.markList(),
      amp: 0,
      period: 8,
      phaseShift: 0,
    };
    if (pose === "up")
      return Object.assign(shared, {
        tilt: (Math.random() * 2 - 1) * 0.1,
        drop: 0.12,
        settle: 0,
        stance: config.skill.stanceMeters,
      });
    const sign = pose === "prone" ? -1 : 1;
    return Object.assign(shared, {
      tilt: sign * (1.4 + Math.random() * 0.2),
      drop: 0.04,
      settle: 40,
    });
  }

  resetEpisode() {
    if (this.episode) this.episode.cleanup();
    if (this.drag) {
      b2DestroyJoint(this.drag.id);
      this.drag = null;
    }
    this.world.rebuild();
    const options = this.episodeOptions();
    this.episode = new this.sim.Episode(
      this.world,
      this.watchedWeights(),
      options.tilt,
      options.drop,
      options,
    );
    this.episode.free = true;
    this.hold = 0;
    if (this.$("graphDialog").open) {
      this.drawNetworkGraph();
      this.paintNetList();
    }
  }

  paintScore(scored) {
    this.ensureScore(scored);
    const total = this.$("total");
    total.className = scored.total >= 0 ? "good" : "bad";
    this.totalValue.textContent = scored.total.toFixed(2);
    this.totalMax.textContent = "max " + scored.max.toFixed(2);
    const terms = scored.terms;
    for (let i = 0; i < terms.length; i++) {
      const value = terms[i].v;
      const row = this.scoreRows[i];
      const width = Math.min(48, (Math.abs(value) / 1.2) * 48);
      const left = value >= 0 ? 50 : 50 - width;
      row.mark.style.left = left + "%";
      row.mark.style.width = width + "%";
      row.mark.style.background = value >= 0 ? "#3ecf8e" : "#ff4d5a";
      row.value.className = value >= 0 ? "good" : "bad";
      row.value.textContent = (value >= 0 ? "+" : "") + value.toFixed(2);
    }
  }

  ensureScore(scored) {
    if (this.scoreRows) return;
    const total = this.$("total");
    total.textContent = "";
    const value = document.createElement("span");
    const max = document.createElement("small");
    max.title = "The ceiling if every row is at its best.";
    total.append(value, max);
    this.totalValue = value;
    this.totalMax = max;
    const root = this.$("terms");
    root.textContent = "";
    this.scoreRows = [];
    const terms = scored.terms;
    for (let i = 0; i < terms.length; i++) {
      const term = terms[i];
      const row = document.createElement("div");
      row.className = "row";
      row.title = term.tip;
      const name = document.createElement("span");
      name.textContent = term.n;
      const bar = document.createElement("span");
      bar.className = "bar";
      const mark = document.createElement("i");
      bar.appendChild(mark);
      const number = document.createElement("span");
      row.append(name, bar, number);
      root.appendChild(row);
      this.scoreRows.push({ mark, value: number });
    }
  }

  paintNetList() {
    const input = this.episode && this.episode.input;
    const hidden = this.episode && this.episode.hidden;
    const action = this.episode && this.episode.action;
    const root = this.$("netlist");
    if (!input || !hidden || !action) {
      root.textContent = "";
      this.netRows = null;
      return;
    }
    if (
      !this.netRows ||
      this.netRows.length !== input.length + hidden.length + action.length
    ) {
      root.textContent = "";
      this.netRows = [];
      const addSection = (title) => {
        const heading = document.createElement("h3");
        heading.textContent = title;
        root.appendChild(heading);
      };
      const addRow = (name, tip) => {
        const row = document.createElement("div");
        row.className = "row";
        row.title = tip;
        const label = document.createElement("span");
        label.textContent = name;
        const bar = document.createElement("span");
        bar.className = "bar";
        const mark = document.createElement("i");
        bar.appendChild(mark);
        const value = document.createElement("span");
        row.append(label, bar, value);
        root.appendChild(row);
        this.netRows.push({ mark, value });
      };
      addSection("Inputs (" + input.length + ")");
      for (let i = 0; i < input.length; i++) {
        const rowLabel = this.observationLabel(i);
        addRow(rowLabel.name, rowLabel.tip);
      }
      addSection("Hidden (" + hidden.length + ")");
      for (let unit = 0; unit < hidden.length; unit++)
        addRow(
          String(unit),
          "Hidden unit " +
            unit +
            ". It keeps a fraction of its previous value.",
        );
      addSection("Outputs");
      for (let joint = 0; joint < action.length; joint++)
        addRow(
          jointNames[joint],
          "Command to " +
            jointNames[joint] +
            ". Positive and negative bend opposite ways.",
        );
    }
    let row = 0;
    const paint = (number) => {
      const item = this.netRows[row++];
      const width = Math.min(100, Math.abs(number) * 100);
      item.mark.style.width = width + "%";
      item.mark.style.background = number >= 0 ? "#3ecf8e" : "#ff4d5a";
      item.value.textContent = (number >= 0 ? "+" : "") + number.toFixed(2);
    };
    for (let i = 0; i < input.length; i++) paint(input[i]);
    for (let unit = 0; unit < hidden.length; unit++) paint(hidden[unit]);
    for (let joint = 0; joint < action.length; joint++) paint(action[joint]);
  }

  observationLabel(index) {
    return (
      observationLabels[index] || {
        name: "input " + index,
        tip: "Observation " + index + ".",
      }
    );
  }

  escapeTip(text) {
    return String(text)
      .replace(/&/g, "&amp;")
      .replace(/"/g, "&quot;")
      .replace(/</g, "&lt;");
  }

  paintPause() {
    this.$("pause").textContent = this.paused ? "Start training" : "Pause";
    this.$("pause").title = this.paused
      ? "Starts the search. The doll on screen keeps moving. This choice is remembered when you reopen the page."
      : "Stops the search. The doll on screen keeps moving. This choice is remembered when you reopen the page.";
  }

  screenPoint(event) {
    const rect = this.canvas.getBoundingClientRect();
    const cursor = this.cursor;
    cursor.x = event.clientX - rect.left;
    cursor.y = event.clientY - rect.top;
    return cursor;
  }

  worldPoint(screenX, screenY, scale, groundY) {
    const pick = this.pick;
    pick.x = (screenX - this.app.screen.width / 2) / scale + this.view.cx;
    pick.y = (groundY - screenY) / scale;
    return pick;
  }

  release() {
    if (this.targetDrag) {
      this.targetDrag = null;
      this.postTargets();
      this.saveUi();
      return;
    }
    if (this.drag) {
      b2DestroyJoint(this.drag.id);
      this.drag = null;
    }
    if (this.aim) {
      const scale = this.view.scale || 100;
      const groundY = this.view.groundY || this.app.screen.height * 0.78;
      const start = this.worldPoint(this.aim.x0, this.aim.y0, scale, groundY);
      let vx = ((this.aim.x - this.aim.x0) / scale) * 5;
      let vy = (-(this.aim.y - this.aim.y0) / scale) * 5;
      const speed = Math.hypot(vx, vy);
      if (speed > 25) {
        vx *= 25 / speed;
        vy *= 25 / speed;
      }
      const kind = thrownKinds[+this.$("objectKind").value];
      const id = this.sim.spawnThrown(
        this.world,
        kind,
        start.x,
        Math.max(start.y, 0.2),
        vx,
        vy,
      );
      this.objects.push({ id, kind });
      if (this.objects.length > 15)
        this.sim.destroyThrown(this.world, this.objects.shift().id);
      this.aim = null;
    }
  }

  mutationStatus(joiner) {
    const boost = this.state.mutationBoost || 1;
    if (!(boost > 1)) return "";
    const pct = Math.round((this.state.mutationFraction || 0) * 100);
    return (
      (joiner || " · ") + "mut x" + boost + (pct > 0 ? " · " + pct + "%" : "")
    );
  }

  chartY(value, originY, plot, drawLow, drawHigh, span) {
    const clamped = Math.min(drawHigh, Math.max(drawLow, value));
    return originY + plot - 4 - ((clamped - drawLow) / span) * (plot - 8);
  }

  drawChart() {
    const canvas = this.$("trainChart");
    const width = canvas.clientWidth;
    const height = canvas.clientHeight;
    if (width < 8 || height < 8) return;
    const ratio = window.devicePixelRatio || 1;
    const pixelsW = Math.floor(width * ratio);
    const pixelsH = Math.floor(height * ratio);
    if (canvas.width !== pixelsW || canvas.height !== pixelsH) {
      canvas.width = pixelsW;
      canvas.height = pixelsH;
    }
    const ctx = canvas.getContext("2d");
    ctx.setTransform(ratio, 0, 0, ratio, 0, 0);
    ctx.clearRect(0, 0, width, height);
    ctx.fillStyle = "#12162a";
    ctx.fillRect(0, 0, width, height);
    const plot = height - 22;
    const history = this.state.hist;
    let drawLow = Infinity;
    let drawHigh = -Infinity;
    for (const sample of history) {
      if (Number.isFinite(sample.mean)) {
        drawLow = Math.min(drawLow, sample.mean);
        drawHigh = Math.max(drawHigh, sample.mean);
      }
      if (Number.isFinite(sample.best)) {
        drawLow = Math.min(drawLow, sample.best);
        drawHigh = Math.max(drawHigh, sample.best);
      }
    }
    if (!Number.isFinite(drawLow)) {
      drawLow = -1;
      drawHigh = 1;
    } else if (drawHigh - drawLow < 1e-6) {
      drawLow -= 1;
      drawHigh += 1;
    } else {
      const pad = (drawHigh - drawLow) * 0.08;
      drawLow -= pad;
      drawHigh += pad;
    }
    const span = drawHigh - drawLow;
    const smoothWindow = config.search.recordWindow;
    const count = history.length;
    let smoothed = this.smoothBuf;
    if (!smoothed || smoothed.length < count) {
      smoothed = new Float64Array(Math.max(count, 400));
      this.smoothBuf = smoothed;
    }
    for (let index = 0; index < count; index++) {
      const begin = Math.max(0, index - smoothWindow + 1);
      let sum = 0;
      let n = 0;
      for (let i = begin; i <= index; i++) {
        if (!Number.isFinite(history[i].mean)) continue;
        sum += history[i].mean;
        n++;
      }
      smoothed[index] = n ? sum / n : NaN;
    }
    const strokeLine = (color, yAt) => {
      ctx.beginPath();
      ctx.strokeStyle = color;
      ctx.lineWidth = 1.5;
      for (let index = 0; index < count; index++) {
        const x = 4 + (index / (count - 1)) * (width - 8);
        const y = this.chartY(yAt(index), 0, plot, drawLow, drawHigh, span);
        if (index) ctx.lineTo(x, y);
        else ctx.moveTo(x, y);
      }
      ctx.stroke();
    };
    if (count >= 2) {
      strokeLine("#6f95ff", (index) =>
        Number.isFinite(smoothed[index]) ? smoothed[index] : drawLow,
      );
      strokeLine("#43d9ad", (index) =>
        Number.isFinite(history[index].best) ? history[index].best : drawLow,
      );
    }
    const last = history[history.length - 1];
    const lastSmooth = count ? smoothed[count - 1] : NaN;
    ctx.font = "11px system-ui, sans-serif";
    ctx.textAlign = "right";
    ctx.textBaseline = "top";
    ctx.fillStyle = "#b7bdd4";
    ctx.fillText(drawLow.toFixed(0) + "–" + drawHigh.toFixed(0), width - 6, 4);
    const legendY = plot + 4;
    ctx.textAlign = "left";
    ctx.fillStyle = "#6f95ff";
    ctx.fillRect(8, legendY + 2, 8, 8);
    ctx.fillText(
      "avg" + (Number.isFinite(lastSmooth) ? " " + lastSmooth.toFixed(2) : ""),
      20,
      legendY,
    );
    ctx.fillStyle = "#43d9ad";
    ctx.fillRect(width / 2, legendY + 2, 8, 8);
    ctx.fillText(
      "gen best" + (last ? " " + last.best.toFixed(2) : ""),
      width / 2 + 12,
      legendY,
    );
  }

  graphSlotY(index, count, top, bottom) {
    if (count <= 1) return (top + bottom) / 2;
    return top + (index * (bottom - top)) / (count - 1);
  }

  ensureGraphBuffers(feedCount, arcCount) {
    if (
      this.feedX0 &&
      this.feedX0.length >= feedCount &&
      this.arcX &&
      this.arcX.length >= arcCount
    )
      return;
    this.feedX0 = new Float32Array(feedCount);
    this.feedY0 = new Float32Array(feedCount);
    this.feedX1 = new Float32Array(feedCount);
    this.feedY1 = new Float32Array(feedCount);
    this.feedBucket = new Uint8Array(feedCount);
    this.feedSign = new Uint8Array(feedCount);
    this.arcX = new Float32Array(arcCount);
    this.arcY0 = new Float32Array(arcCount);
    this.arcY1 = new Float32Array(arcCount);
    this.arcBucket = new Uint8Array(arcCount);
    this.arcSign = new Uint8Array(arcCount);
    this.arcSelf = new Uint8Array(arcCount);
  }

  // Faint under 1, full alpha at 1e8. tanh saturates the units, so the log of |w|
  // is what still separates a huge weight.
  weightTone(mag) {
    if (!(mag > 0)) return 0;
    if (mag <= 1) return mag * 0.28;
    return 0.28 + 0.72 * Math.min(1, Math.log10(mag) / 8);
  }

  graphMarks(weight, source) {
    const mag = Math.abs(weight);
    const live = Math.max(Math.abs(source || 0), 0.35);
    const drive = Math.min(1, live * this.weightTone(mag));
    return Math.min(5, (drive * 6) | 0);
  }

  pushWire(weight, source, x0, y0, x1, y1) {
    const at = this.feedUsed;
    this.feedX0[at] = x0;
    this.feedY0[at] = y0;
    this.feedX1[at] = x1;
    this.feedY1[at] = y1;
    this.feedBucket[at] = this.graphMarks(weight, source);
    this.feedSign[at] = weight >= 0 ? 1 : 0;
    this.feedUsed = at + 1;
  }

  pushArc(weight, source, x, y0, y1, self) {
    const at = this.arcUsed;
    this.arcX[at] = x;
    this.arcY0[at] = y0;
    this.arcY1[at] = y1;
    this.arcBucket[at] = this.graphMarks(weight, source);
    this.arcSign[at] = weight >= 0 ? 1 : 0;
    this.arcSelf[at] = self ? 1 : 0;
    this.arcUsed = at + 1;
  }

  strokeGraph(context) {
    context.lineWidth = 1;
    context.setLineDash([]);
    const feedUsed = this.feedUsed;
    for (let bucket = 0; bucket < 6; bucket++) {
      context.beginPath();
      for (let i = 0; i < feedUsed; i++) {
        if (this.feedBucket[i] !== bucket || !this.feedSign[i]) continue;
        context.moveTo(this.feedX0[i], this.feedY0[i]);
        context.lineTo(this.feedX1[i], this.feedY1[i]);
      }
      context.strokeStyle = "rgba(62,207,142," + GRAPH_ALPHA[bucket] + ")";
      context.stroke();
      context.beginPath();
      for (let i = 0; i < feedUsed; i++) {
        if (this.feedBucket[i] !== bucket || this.feedSign[i]) continue;
        context.moveTo(this.feedX0[i], this.feedY0[i]);
        context.lineTo(this.feedX1[i], this.feedY1[i]);
      }
      context.strokeStyle = "rgba(255,77,90," + GRAPH_ALPHA[bucket] + ")";
      context.stroke();
    }
    context.setLineDash([4, 3]);
    const arcUsed = this.arcUsed;
    for (let bucket = 0; bucket < 6; bucket++) {
      context.beginPath();
      for (let i = 0; i < arcUsed; i++) {
        if (this.arcBucket[i] !== bucket || !this.arcSign[i]) continue;
        this.strokeArc(context, i);
      }
      context.strokeStyle = "rgba(62,207,142," + GRAPH_ALPHA[bucket] + ")";
      context.stroke();
      context.beginPath();
      for (let i = 0; i < arcUsed; i++) {
        if (this.arcBucket[i] !== bucket || this.arcSign[i]) continue;
        this.strokeArc(context, i);
      }
      context.strokeStyle = "rgba(255,77,90," + GRAPH_ALPHA[bucket] + ")";
      context.stroke();
    }
    context.setLineDash([]);
  }

  strokeArc(context, index) {
    const x = this.arcX[index];
    const y0 = this.arcY0[index];
    const y1 = this.arcY1[index];
    if (this.arcSelf[index]) {
      const start = 0.7;
      context.moveTo(x - 14 + Math.cos(start) * 6, y1 + Math.sin(start) * 6);
      context.arc(x - 14, y1, 6, start, Math.PI * 2.15);
    } else {
      context.moveTo(x, y0);
      context.quadraticCurveTo(x - 36, (y0 + y1) / 2, x, y1);
    }
  }

  drawNetworkGraph() {
    const canvas = this.$("graphCanvas");
    const missing = this.$("graphMissing");
    const episode = this.episode;
    const weights = episode && episode.weights;
    let maxAbs = 0;
    if (weights) {
      for (let i = 0; i < weights.length; i++) {
        const mag = Math.abs(weights[i]);
        if (mag > maxAbs) maxAbs = mag;
      }
    }
    this.$("graphTitle").textContent =
      this.networkTitle() +
      (maxAbs ? "   |w|max " + maxAbs.toExponential(2) : "");
    const layout = ElmanNetwork.layout;
    const context = canvas.getContext("2d");
    if (!weights || weights.length !== ElmanNetwork.weightCount) {
      missing.textContent = "No weights to draw.";
      context.setTransform(1, 0, 0, 1, 0, 0);
      context.clearRect(0, 0, canvas.width, canvas.height);
      return;
    }
    missing.textContent = "";
    const feedforward = layout.feedforward;
    const recurrent = layout.recurrent;
    const feedbackCount = layout.feedback;
    const inputs = episode.input;
    const firstAct = episode.network && episode.network.compressed;
    const feedbackAct = episode.network && episode.network.feedback;
    const feedbackFrom = episode.network && episode.network.feedbackFrom;
    const memoryAct = episode.hidden;
    const outputs = episode.action;
    const joints = jointNames.length;
    const width = Math.max(320, canvas.clientWidth || 640);
    const height = Math.max(360, canvas.clientHeight || 480);
    const pixelRatio = Math.min(1.5, devicePixelRatio || 1);
    if (
      canvas.width !== Math.floor(width * pixelRatio) ||
      canvas.height !== Math.floor(height * pixelRatio)
    ) {
      canvas.width = Math.floor(width * pixelRatio);
      canvas.height = Math.floor(height * pixelRatio);
    }
    context.setTransform(pixelRatio, 0, 0, pixelRatio, 0, 0);
    context.clearRect(0, 0, width, height);
    context.font = "11px system-ui, sans-serif";
    context.textBaseline = "middle";
    if (!this.drawNetworkGraph.labelPad) {
      let widest = 72;
      for (let i = 0; i < config.net.inputCount; i++)
        widest = Math.max(
          widest,
          context.measureText(this.observationLabel(i).name).width,
        );
      this.drawNetworkGraph.labelPad = Math.ceil(widest + 16);
    }
    const left = Math.min(this.drawNetworkGraph.labelPad, width * 0.32);
    const right = width - 88;
    const top = 28;
    const bottom = height - 12;
    if (!this.graphColumns) {
      const built = [
        {
          title: "",
          count: config.net.inputCount,
          radius: 1.6,
          align: "right",
          tag: (i) => this.observationLabel(i).name,
        },
      ];
      if (feedforward)
        built.push({
          title: "hidden 1",
          count: feedforward,
          radius: 3.5,
          align: "left",
          tag: (i) => "a" + i,
        });
      if (feedbackCount)
        built.push({
          title: "feedback",
          count: feedbackCount,
          radius: 3.5,
          align: "left",
          tag: (i) => "f" + i,
        });
      built.push({
        title: feedforward ? "hidden 2" : "recurrent dashed",
        count: recurrent,
        radius: feedforward ? 3.5 : 5,
        align: "left",
        tag: (i) => (feedforward ? "b" : "h") + i,
      });
      built.push({
        title: "",
        count: joints,
        x: right,
        radius: 5,
        align: "left",
        tag: (i) => jointNames[i],
      });
      this.graphColumns = built;
    }
    const columns = this.graphColumns;
    columns[0].x = left;
    columns[0].values = inputs;
    let columnCursor = 1;
    if (feedforward) columns[columnCursor++].values = firstAct;
    if (feedbackCount) columns[columnCursor++].values = feedbackAct;
    columns[columnCursor++].values = memoryAct;
    columns[columnCursor].x = right;
    columns[columnCursor].values = outputs;
    const denom = columns.length - 1;
    for (let i = 1; i < columns.length - 1; i++)
      columns[i].x = left + ((right - left) * i) / denom;
    const yAt = (column, index) =>
      this.graphSlotY(index, column.count, top, bottom);
    const inputCount = config.net.inputCount;
    const feedCount =
      (feedforward ? feedforward * inputCount : 0) +
      feedbackCount * joints +
      recurrent * (feedforward ? feedforward : inputCount) +
      recurrent * feedbackCount +
      joints * recurrent;
    this.ensureGraphBuffers(feedCount, recurrent * recurrent);
    this.feedUsed = 0;
    this.arcUsed = 0;
    const inputColumn = columns[0];
    const jointColumn = columns[columns.length - 1];
    const memoryColumn = columns[columns.length - 2];
    const firstColumn = feedforward ? columns[1] : null;
    const feedbackColumn = feedbackCount ? columns[feedforward ? 2 : 1] : null;
    if (feedforward) {
      for (let unit = 0; unit < feedforward; unit++) {
        const base = unit * layout.feedforwardStride;
        const hiddenY = yAt(firstColumn, unit);
        for (let i = 0; i < config.net.inputCount; i++) {
          this.pushWire(
            weights[base + 1 + i],
            inputs && inputs[i],
            inputColumn.x,
            yAt(inputColumn, i),
            firstColumn.x,
            hiddenY,
          );
        }
      }
    }
    if (feedbackCount) {
      for (let unit = 0; unit < feedbackCount; unit++) {
        const base = layout.feedbackBase + unit * layout.feedbackStride;
        const feedbackY = yAt(feedbackColumn, unit);
        for (let joint = 0; joint < joints; joint++) {
          this.pushWire(
            weights[base + 1 + joint],
            feedbackFrom && feedbackFrom[joint],
            jointColumn.x,
            yAt(jointColumn, joint),
            feedbackColumn.x,
            feedbackY,
          );
        }
      }
    }
    for (let unit = 0; unit < recurrent; unit++) {
      const base = layout.recurrentBase + unit * layout.recurrentStride;
      const hiddenY = yAt(memoryColumn, unit);
      let offset = 1;
      if (feedforward) {
        for (let i = 0; i < feedforward; i++) {
          this.pushWire(
            weights[base + offset + i],
            firstAct && firstAct[i],
            firstColumn.x,
            yAt(firstColumn, i),
            memoryColumn.x,
            hiddenY,
          );
        }
        offset += feedforward;
      } else {
        for (let i = 0; i < config.net.inputCount; i++) {
          this.pushWire(
            weights[base + offset + i],
            inputs && inputs[i],
            inputColumn.x,
            yAt(inputColumn, i),
            memoryColumn.x,
            hiddenY,
          );
        }
        offset += config.net.inputCount;
      }
      for (let i = 0; i < feedbackCount; i++) {
        this.pushWire(
          weights[base + offset + i],
          feedbackAct && feedbackAct[i],
          feedbackColumn.x,
          yAt(feedbackColumn, i),
          memoryColumn.x,
          hiddenY,
        );
      }
      offset += feedbackCount;
      for (let other = 0; other < recurrent; other++) {
        this.pushArc(
          weights[base + offset + other],
          memoryAct && memoryAct[other],
          memoryColumn.x,
          yAt(memoryColumn, other),
          hiddenY,
          other === unit,
        );
      }
    }
    for (let joint = 0; joint < joints; joint++) {
      const base = layout.outputBase + joint * (recurrent + 1);
      for (let unit = 0; unit < recurrent; unit++) {
        this.pushWire(
          weights[base + 1 + unit],
          memoryAct && memoryAct[unit],
          memoryColumn.x,
          yAt(memoryColumn, unit),
          jointColumn.x,
          yAt(jointColumn, joint),
        );
      }
    }
    this.strokeGraph(context);
    const fillOf = (value) => {
      const mag = Math.min(1, Math.abs(value || 0));
      const alpha = 0.18 + 0.82 * mag;
      return (value || 0) >= 0
        ? "rgba(62,207,142," + alpha + ")"
        : "rgba(255,77,90," + alpha + ")";
    };
    const dot = (x, y, radius, value) => {
      context.fillStyle = fillOf(value);
      context.beginPath();
      context.arc(x, y, radius, 0, Math.PI * 2);
      context.fill();
    };
    for (const column of columns) {
      for (let index = 0; index < column.count; index++)
        dot(
          column.x,
          yAt(column, index),
          column.radius,
          column.values && column.values[index],
        );
    }
    const tag = (text, x, y, align) => {
      context.textAlign = align;
      const box = context.measureText(text).width;
      const boxX =
        align === "right" ? x - box : align === "center" ? x - box / 2 : x;
      context.fillStyle = "#12162a";
      context.fillRect(boxX - 2, y - 7, box + 4, 14);
      context.fillStyle = "#e7e9f5";
      context.fillText(text, x, y);
    };
    for (const column of columns)
      if (column.title) tag(column.title, column.x, 11, "center");
    const labeled = this.graphLabeled || (this.graphLabeled = []);
    const usedY = this.graphUsedY || (this.graphUsedY = []);
    labeled.length = 0;
    usedY.length = 0;
    const place = (index) => {
      const y = yAt(inputColumn, index);
      for (let k = 0; k < usedY.length; k++)
        if (Math.abs(usedY[k] - y) < 12) return;
      usedY.push(y);
      labeled.push(index);
    };
    const labelStep = Math.max(
      4,
      Math.ceil(config.net.inputCount / Math.max(6, ((bottom - top) / 15) | 0)),
    );
    for (let i = 0; i < config.net.inputCount; i += labelStep) place(i);
    place(config.net.inputCount - 1);
    if (inputs && inputs.length) {
      let rank = this.graphRank;
      if (!rank || rank.length !== config.net.inputCount) {
        rank = [];
        for (let i = 0; i < config.net.inputCount; i++) rank.push(i);
        this.graphRank = rank;
        const self = this;
        this.compareAbsInput = (a, b) =>
          Math.abs(self.graphInputs[b] || 0) -
          Math.abs(self.graphInputs[a] || 0);
      } else {
        for (let i = 0; i < rank.length; i++) rank[i] = i;
      }
      this.graphInputs = inputs;
      rank.sort(this.compareAbsInput);
      for (let k = 0; k < 8 && k < rank.length; k++) place(rank[k]);
    }
    for (const index of labeled)
      tag(
        inputColumn.tag(index),
        inputColumn.x - 8,
        yAt(inputColumn, index),
        "right",
      );
    const seen = this.graphSeen || (this.graphSeen = []);
    for (let columnIndex = 1; columnIndex < columns.length; columnIndex++) {
      const column = columns[columnIndex];
      seen.length = 0;
      const step = Math.max(
        1,
        Math.ceil(column.count / Math.max(6, (bottom - top) / 15)),
      );
      const keep = (index) => {
        const y = yAt(column, index);
        for (let k = 0; k < seen.length; k++)
          if (Math.abs(seen[k] - y) < 12) return;
        seen.push(y);
        const gap = column.align === "right" ? -8 : 8;
        tag(column.tag(index), column.x + gap, y, column.align);
      };
      for (let index = 0; index < column.count; index += step) keep(index);
      keep(column.count - 1);
    }
  }

  fileStamp(date) {
    const pad = (n) => String(n).padStart(2, "0");
    return (
      date.getFullYear() +
      pad(date.getMonth() + 1) +
      pad(date.getDate()) +
      "-" +
      pad(date.getHours()) +
      pad(date.getMinutes()) +
      pad(date.getSeconds())
    );
  }

  async thetaHash8(weights) {
    const packed = JSON.stringify(quantizeWeights(weights));
    const digest = await crypto.subtle.digest(
      "SHA-256",
      new TextEncoder().encode(packed),
    );
    return Array.from(new Uint8Array(digest), (byte) =>
      byte.toString(16).padStart(2, "0"),
    )
      .join("")
      .slice(0, 8);
  }

  async freeStem(hash) {
    let when = new Date();
    for (let n = 0; n < 5; n++) {
      const stem = this.fileStamp(when) + "_" + hash;
      try {
        await this.backupDir.getFileHandle(stem + ".json");
      } catch (error) {
        return stem;
      }
      when = new Date(when.getTime() + 1000);
    }
    return this.fileStamp(new Date()) + "_" + hash;
  }

  backupPayload(role, stem) {
    return {
      role,
      name: stem,
      gen: this.state.gen,
      bestEverScore: this.state.bestEverScore,
      bestEverGen: this.state.bestEverGen || 0,
      bestEverName: role === "best-ever" ? stem : this.state.bestEverName || "",
      trainingPhase: config.curriculum.phase,
      inputCount: config.net.inputCount,
      theta: quantizeWeights(this.state.theta),
      bestEverTheta: quantizeWeights(this.state.bestEverTheta),
    };
  }

  async writeBackup(role) {
    const weights =
      role === "best-ever" ? this.state.bestEverTheta : this.state.theta;
    if (!weights || !this.backupDir) return "";
    try {
      const hash = await this.thetaHash8(weights);
      const stem = await this.freeStem(hash);
      const file = stem + ".json";
      const payload = this.backupPayload(role, stem);
      payload.file = file;
      const handle = await this.backupDir.getFileHandle(file, { create: true });
      const writable = await handle.createWritable();
      await writable.write(JSON.stringify(payload));
      await writable.close();
      this.logEvent("JSON saved (" + file + ")");
      return stem;
    } catch (error) {
      return "";
    }
  }

  async rememberBest() {
    const every = +this.$("snapshotEvery").value || 100;
    if (!this.state.gen || !this.state.bestEverTheta) return;
    if (this.state.bestEverScore > this.backedBestScore) {
      this.backedBestScore = this.state.bestEverScore;
      this.state.bestEverGen = this.state.gen;
      this.state.bestEverName = "";
      const stem = await this.writeBackup("best-ever");
      if (stem) this.state.bestEverName = stem;
    }
    if (this.state.gen % every !== 0) return;
    const parentStem = await this.writeBackup("parent");
    if (parentStem) this.visit.parentName = parentStem;
    if (!this.state.snapshots) this.state.snapshots = [];
    const previous = this.state.snapshots[this.state.snapshots.length - 1];
    if (previous && !(this.state.bestEverScore > previous.score)) return;
    const weights = Float32Array.from(this.state.bestEverTheta);
    this.state.snapshots.push({
      gen: this.state.gen,
      score: this.state.bestEverScore,
      weights,
      name: this.state.bestEverName || "",
    });
    if (this.state.snapshots.length > 40) this.state.snapshots.shift();
    this.fillSavedList();
    await this.syncSnapshotStore();
  }

  fillSavedList() {
    const select = this.$("savedNet");
    const current = select.value;
    select.innerHTML = "";
    const blank = document.createElement("option");
    blank.value = "";
    blank.textContent = "—";
    select.appendChild(blank);
    for (const item of this.state.snapshots || []) {
      const option = document.createElement("option");
      option.value = String(item.gen);
      option.textContent =
        "gen " + item.gen + " · " + Number(item.score).toFixed(2);
      select.appendChild(option);
    }
    const keep = [...select.options].some((option) => option.value === current);
    select.value = keep ? current : "";
    if (
      current &&
      !keep &&
      !this.$("watchParent").checked &&
      !this.$("watchGeneration").checked &&
      !this.$("watchEver").checked
    ) {
      this.$("watchParent").checked = true;
    }
  }

  sessionRecord() {
    const state = this.state;
    const hist = state.hist || [];
    const start = Math.max(0, hist.length - 80);
    const kept = [];
    for (let i = start; i < hist.length; i++) kept.push(hist[i]);
    return {
      gen: state.gen,
      inputCount: config.net.inputCount,
      weightCount: ElmanNetwork.weightCount,
      theta: state.theta,
      bestGenTheta: state.bestGenTheta,
      bestEverTheta: state.bestEverTheta,
      m: state.m,
      v: state.v,
      k: state.k || 0,
      bestEverScore: state.bestEverScore,
      bestEverName: state.bestEverName || "",
      bestEverGen: Number.isFinite(state.bestEverGen)
        ? state.bestEverGen
        : null,
      networkName: state.networkName || "",
      eliteScore: state.eliteScore,
      parentMoves: state.parentMoves || 0,
      parentMoveNote: state.parentMoveNote || "",
      flatGens: state.flatGens || 0,
      bestSmooth: Number.isFinite(state.bestSmooth) ? state.bestSmooth : null,
      hist: kept,
    };
  }

  saveSession() {
    if (!storeReady() || !this.state) return;
    this.sessionDirty = true;
    if (this.sessionWriting) return;
    this.flushSession();
  }

  flushSession() {
    if (!this.sessionDirty || this.sessionWriting || !storeReady()) return;
    this.sessionDirty = false;
    this.sessionWriting = true;
    const started = performance.now();
    let pending;
    try {
      pending = putSession(this.sessionRecord());
    } catch (error) {
      this.sessionWriting = false;
      return;
    }
    this.saveMs = performance.now() - started;
    pending
      .then(
        () => {},
        () => {},
      )
      .then(() => {
        this.sessionWriting = false;
        if (this.sessionDirty) this.flushSession();
      });
  }

  syncSnapshotStore() {
    if (!storeReady() || !this.state) return Promise.resolve();
    return syncSnapshots(this.state.snapshots || []).then(
      () => {},
      () => {},
    );
  }

  clearSnapshots() {
    return this.syncSnapshotStore();
  }

  async openSession() {
    try {
      await openStore();
    } catch (error) {
      return new EvolutionSearch(this.randomNormal);
    }
    let stored = null;
    try {
      stored = await loadStored();
    } catch (error) {
      stored = null;
    }
    if (stored && stored.session && stored.session.theta) {
      const revived = this.stateFromPacket(stored.session, stored.snapshots);
      if (revived) return revived;
      this.discardedNetwork = true;
    }
    let legacy = null;
    try {
      legacy = JSON.parse(localStorage.getItem("ragdoll-panel") || "null");
    } catch (error) {
      legacy = null;
    }
    if (legacy && legacy.theta) {
      const revived = this.stateFromPacket(legacy, legacy.snapshots);
      if (revived) {
        localStorage.removeItem("ragdoll-panel");
        this.state = revived;
        this.saveSession();
        await this.syncSnapshotStore();
        return revived;
      }
      this.discardedNetwork = true;
    }
    return new EvolutionSearch(this.randomNormal);
  }

  toast(message) {
    const el = this.$("toast");
    if (!el) return;
    el.hidden = false;
    el.textContent = message;
    clearTimeout(this.toastTimer);
    this.toastTimer = setTimeout(() => {
      el.hidden = true;
    }, 5000);
  }

  freshSearch() {
    this.random = makeSeededRandom(Date.now() & 0xffff);
    this.sim.setRandom(Date.now() & 0xffff);
    this.state = new EvolutionSearch(this.randomNormal);
    this.state.snapshots = [];
    this.backedBestScore = this.state.bestEverScore ?? -1e9;
    this.visit.parentName = "";
    this.visit.loadedLabel = "";
    this.visit.trained = false;
    this.fillSavedList();
    this.clearSnapshots();
    this.saveSession();
    this.pushSearch();
    this.resetEpisode();
    this.saveUi();
  }

  takeWeights(source) {
    const length = ElmanNetwork.weightCount;
    if (source instanceof Float32Array && source.length === length)
      return source;
    return ElmanNetwork.copy(source);
  }

  saveUi() {
    try {
      const watch = this.$("watchEver").checked
        ? "ever"
        : this.$("watchGeneration").checked
          ? "gen"
          : "parent";
      localStorage.setItem(
        "ragdoll-ui",
        JSON.stringify({
          paused: this.paused,
          snapshotEvery: this.$("snapshotEvery").value,
          watch,
          savedNet: this.$("savedNet").value,
          startPose: this.$("startPose").value,
          viewRays: this.$("viewRays").checked,
          viewColliders: this.$("viewColliders").checked,
          passive: this.$("passive").checked,
          objectKind: this.$("objectKind").value,
          speed: this.$("speed").value,
          phaseStartGen: this.phaseStartGen,
          curriculumPhase: config.curriculum.phase,
          skillLevel: this.$("skillLevel").value,
          showTrainPose: this.$("showTrainPose").checked,
          playMode: this.trainMode ? "train" : "sandbox",
          jitter: this.jitter,
          randomStart: this.randomStart,
          trainSaved: this.trainSaved,
          targets: this.targets.map((item) => ({
            on: !!item.on,
            x: item.x,
            y: item.y,
            angle: item.angle,
            placed: !!item.placed,
          })),
          trainMarks: this.trainMarks.map((item) => ({
            on: !!item.on,
            x: item.x,
            y: item.y,
            angle: item.angle,
            placed: !!item.placed,
          })),
        }),
      );
    } catch (error) {
      /* settings are small; a quota error still leaves the run going */
    }
  }

  restoreUi() {
    let saved;
    try {
      saved = JSON.parse(localStorage.getItem("ragdoll-ui") || "null");
    } catch (error) {
      return;
    }
    if (!saved) return;
    const pick = (id, value) => {
      if (value == null || value === "") return;
      const el = this.$(id);
      const text = String(value);
      if (
        el.options &&
        ![...el.options].some((option) => option.value === text)
      )
        return;
      el.value = text;
    };
    pick("snapshotEvery", saved.snapshotEvery);
    pick("startPose", saved.startPose);
    pick("objectKind", saved.objectKind);
    if (saved.speed != null) {
      this.$("speed").value = saved.speed;
      this.$("speedValue").textContent = this.$("speed").value + " m/s";
    }
    this.$("viewRays").checked = !!saved.viewRays;
    this.$("viewColliders").checked = !!saved.viewColliders;
    this.$("passive").checked = !!saved.passive;
    if (Array.isArray(saved.targets)) {
      saved.targets.forEach((item, index) => {
        const target = this.targets[index];
        if (!target || !item) return;
        target.on = !!item.on;
        if (Number.isFinite(+item.x)) target.x = +item.x;
        if (Number.isFinite(+item.y)) target.y = +item.y;
        if (Number.isFinite(+item.angle)) target.angle = +item.angle;
        target.placed = !!item.placed;
      });
      this.paintTargetToggles();
      this.syncTrainBoxes();
    }
    this.$("watchParent").checked =
      saved.watch !== "gen" && saved.watch !== "ever";
    this.$("watchGeneration").checked = saved.watch === "gen";
    this.$("watchEver").checked = saved.watch === "ever";
    if (saved.savedNet) {
      const select = this.$("savedNet");
      const text = String(saved.savedNet);
      if ([...select.options].some((option) => option.value === text)) {
        select.value = text;
        this.$("watchParent").checked = false;
        this.$("watchGeneration").checked = false;
        this.$("watchEver").checked = false;
      }
    }
    if (saved.paused === false) this.paused = false;
    pick("skillLevel", saved.skillLevel);
    this.$("showTrainPose").checked = !!saved.showTrainPose;
    if (saved.playMode === "train" || saved.playMode === "sandbox") {
      this.$("playMode").value = saved.playMode;
      this.trainMode = saved.playMode === "train";
    }
    if (Number.isFinite(+saved.jitter)) {
      this.jitter = Math.min(0.2, Math.max(0, +saved.jitter));
      this.$("trainJitter").value = String(this.jitter);
      this.paintJitter();
    }
    if (saved.randomStart != null) {
      this.randomStart = !!saved.randomStart;
      this.$("randomStart").checked = this.randomStart;
    }
    this.trainSaved = !!saved.trainSaved;
    if (Array.isArray(saved.trainMarks)) {
      saved.trainMarks.forEach((item, index) => {
        const target = this.trainMarks[index];
        if (!target || !item) return;
        target.on = !!item.on;
        if (Number.isFinite(+item.x)) target.x = +item.x;
        if (Number.isFinite(+item.y)) target.y = +item.y;
        if (Number.isFinite(+item.angle)) target.angle = +item.angle;
        target.placed = !!item.placed;
      });
      this.trainSaved = true;
    }
    this.paintTargetToggles();
    this.syncTrainBoxes();
  }

  networkFromJson(data) {
    const rawArray = Array.isArray(data);
    const packet = rawArray ? { theta: data } : data;
    if (!packet || typeof packet !== "object") return null;
    const parent = ElmanNetwork.copy(packet.theta);
    if (!parent) return null;
    const revived = new EvolutionSearch(this.randomNormal);
    revived.theta = parent;
    revived.bestEverTheta =
      (!rawArray && ElmanNetwork.copy(packet.bestEverTheta)) || parent.slice();
    revived.bestGenTheta =
      (!rawArray && ElmanNetwork.copy(packet.bestGenTheta)) || parent.slice();
    revived.m = new Float32Array(parent.length);
    revived.v = new Float32Array(parent.length);
    revived.k = 0;
    if (rawArray) return revived;
    if (Number.isFinite(+packet.gen) && +packet.gen >= 0)
      revived.gen = Math.floor(+packet.gen);
    if (Number.isFinite(+packet.bestEverScore))
      revived.bestEverScore = +packet.bestEverScore;
    if (typeof packet.bestEverName === "string")
      revived.bestEverName = packet.bestEverName.replace(/\.json$/i, "");
    if (Number.isFinite(+packet.bestEverGen) && +packet.bestEverGen >= 0)
      revived.bestEverGen = Math.floor(+packet.bestEverGen);
    if (typeof packet.name === "string")
      revived.networkName = packet.name.replace(/\.json$/i, "");
    if (Number.isFinite(+packet.eliteScore))
      revived.eliteScore = +packet.eliteScore;
    if (Number.isFinite(+packet.parentMoves) && +packet.parentMoves > 0)
      revived.parentMoves = Math.floor(+packet.parentMoves);
    if (typeof packet.parentMoveNote === "string")
      revived.parentMoveNote = packet.parentMoveNote;
    if (Number.isFinite(+packet.bestSmooth))
      revived.bestSmooth = +packet.bestSmooth;
    if (Array.isArray(packet.hist)) {
      revived.hist = packet.hist.filter(
        (item) =>
          item && Number.isFinite(+item.mean) && Number.isFinite(+item.best),
      );
    }
    if (Array.isArray(packet.snapshots)) {
      revived.snapshots = packet.snapshots
        .slice(-40)
        .map((item) => ({
          gen: item && item.gen,
          score: item && item.score,
          name:
            item && typeof item.name === "string"
              ? item.name.replace(/\.json$/i, "")
              : "",
          weights: item && ElmanNetwork.copy(item.weights),
        }))
        .filter(
          (item) =>
            item.weights &&
            Number.isFinite(+item.gen) &&
            Number.isFinite(+item.score),
        );
    }
    return revived;
  }

  stateFromPacket(packet, snapshotSource) {
    if (!packet || !packet.theta) return null;
    if (
      packet.weightCount != null &&
      +packet.weightCount !== ElmanNetwork.weightCount
    )
      return null;
    if (
      packet.inputCount != null &&
      +packet.inputCount !== config.net.inputCount
    )
      return null;
    const parent = this.takeWeights(packet.theta);
    if (!parent) return null;
    const revived = new EvolutionSearch(this.randomNormal);
    revived.theta = parent;
    revived.gen = packet.gen || 0;
    revived.bestGenTheta =
      this.takeWeights(packet.bestGenTheta) || parent.slice();
    revived.bestEverTheta =
      this.takeWeights(packet.bestEverTheta) || parent.slice();
    revived.bestEverScore = packet.bestEverScore ?? -1e9;
    revived.bestEverName =
      typeof packet.bestEverName === "string" ? packet.bestEverName : "";
    if (packet.bestEverGen != null && Number.isFinite(+packet.bestEverGen))
      revived.bestEverGen = Math.floor(+packet.bestEverGen);
    revived.networkName =
      typeof packet.networkName === "string" ? packet.networkName : "";
    revived.eliteScore = packet.eliteScore ?? -1e9;
    if (Number.isFinite(+packet.parentMoves) && +packet.parentMoves > 0)
      revived.parentMoves = Math.floor(+packet.parentMoves);
    if (typeof packet.parentMoveNote === "string")
      revived.parentMoveNote = packet.parentMoveNote;
    if (Number.isFinite(+packet.flatGens) && +packet.flatGens > 0)
      revived.flatGens = Math.floor(+packet.flatGens);
    if (Number.isFinite(+packet.bestSmooth))
      revived.bestSmooth = +packet.bestSmooth;
    if (Number.isFinite(+packet.k) && +packet.k >= 0)
      revived.k = Math.floor(+packet.k);
    const moments = this.takeWeights(packet.m);
    const second = this.takeWeights(packet.v);
    if (moments) revived.m = moments;
    if (second) revived.v = second;
    revived.refreshMutationBoost();
    revived.hist = Array.isArray(packet.hist)
      ? packet.hist.filter(
          (item) =>
            item && Number.isFinite(+item.mean) && Number.isFinite(+item.best),
        )
      : [];
    const source = snapshotSource || packet.snapshots || [];
    revived.snapshots = source
      .slice(-40)
      .map((item) => ({
        gen: item.gen,
        score: item.score,
        name: item.name || "",
        weights: item && this.takeWeights(item.weights),
      }))
      .filter(
        (item) =>
          item.weights &&
          Number.isFinite(+item.gen) &&
          Number.isFinite(+item.score),
      );
    revived.snapshots.sort((a, b) => a.gen - b.gen);
    return revived;
  }
}

new NeuralRagDollWorld();
