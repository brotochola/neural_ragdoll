import { CreateMouseJoint, b2MouseJoint_SetTarget, b2DestroyJoint, b2DestroyBody, b2Body_GetPosition, b2Body_GetRotation, b2Rot_GetAngle, b2Vec2 } from "https://cdn.jsdelivr.net/npm/phaser-box2d@1.1.0/dist/PhaserBox2D.min.js";
import * as Box2D from "https://cdn.jsdelivr.net/npm/phaser-box2d@1.1.0/dist/PhaserBox2D.min.js";
import { config, setTrainingPhase, setTrainingByPhases, activeTrainingPhase } from "./config.js";
import { scoreDecision, isStanding } from "./src/reward.js";
import { ElmanNetwork, EvolutionSearch } from "./src/net.js";
import { clamp, makeSeededRandom, randomNormalSample } from "./src/utils.js";
import { createSim, observationLabels, jointNames, thrownKinds } from "./src/sim.js";

export class NeuralRagDollWorld {
  constructor() {
    this.start();
  }

  async start() {
    this.$ = (id) => document.getElementById(id);
    this.sim = createSim(Box2D);
    this.sim.setRandom((Date.now() & 0xffff) || 1);
    this.random = makeSeededRandom(7);
    this.randomNormal = () => randomNormalSample(this.random);
    this.state = this.loadSession() || new EvolutionSearch(this.randomNormal);
    this.paused = true;
    this.phaseStartGen = this.state.gen || 0;
    this.backupDir = null;
    this.backedBestScore = this.state.bestEverScore ?? -1e9;
    this.visit = { log: [], parentName: "", loadedLabel: "", trained: false };
    this.vector = (x, y) => new b2Vec2(x, y);

    for (let i = 0; i < thrownKinds.length; i++) {
      const option = document.createElement("option");
      option.value = String(i);
      option.textContent = thrownKinds[i].name;
      this.$("objectKind").appendChild(option);
    }
    this.fillSavedList();

    this.pool = { workers: [], ready: false };
    this.world = this.sim.build();
    this.episode = undefined;
    this.hold = 0;
    this.objects = [];
    this.drag = null;
    this.aim = null;
    this.view = { cx: 0 };
    this.$("pause").onclick = () => {
      this.paused = !this.paused;
      this.paintPause();
      this.logEvent(this.paused ? "training paused" : "training started");
    };
    this.paintPause();
    this.$("restart").onclick = () => {
      this.random = makeSeededRandom(Date.now() & 0xffff);
      this.sim.setRandom(Date.now() & 0xffff);
      this.state = new EvolutionSearch(this.randomNormal);
      this.state.snapshots = [];
      if (config.curriculum.byPhases) {
        setTrainingPhase(config.curriculum.phases[0].id);
        this.phaseStartGen = 0;
      }
      this.backedBestScore = this.state.bestEverScore ?? -1e9;
      this.visit.parentName = "";
      this.visit.loadedLabel = "";
      this.visit.trained = false;
      this.fillSavedList();
      this.saveSession();
      this.resetEpisode();
      this.logEvent("restart");
    };
    this.$("reset").onclick = () => this.resetEpisode();
    this.$("saveFolder").onclick = async () => {
      if (!window.showDirectoryPicker) {
        this.$("error").textContent = "This browser has no folder picker.";
        return;
      }
      try {
        this.backupDir = await window.showDirectoryPicker({ mode: "readwrite" });
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
      const file = this.$("loadNetworkFile").files && this.$("loadNetworkFile").files[0];
      if (!file) return;
      let data;
      try { data = JSON.parse(await file.text()); }
      catch (error) {
        this.$("error").textContent = "Could not read that network file.";
        this.logEvent("JSON load failed");
        return;
      }
      const revived = this.networkFromJson(data);
      if (!revived) {
        this.$("error").textContent = "That file is not a network this page can load.";
        this.logEvent("JSON load failed");
        return;
      }
      const stem = file.name.replace(/\.json$/i, "");
      let hash = "";
      try { hash = await this.thetaHash8(revived.theta); }
      catch (error) { hash = ""; }
      this.visit.loadedLabel = hash && !stem.endsWith(hash) ? stem + " " + hash : stem;
      this.visit.parentName = "";
      this.visit.trained = false;
      if (hash) revived.networkName = this.fileStamp(new Date()) + "_" + hash;
      if (data && data.role === "best-ever" && !revived.bestEverName) revived.bestEverName = stem;
      this.state = revived;
      this.backedBestScore = this.state.bestEverScore ?? -1e9;
      this.paused = true;
      this.paintPause();
      this.$("error").textContent = "";
      this.saveSession();
      this.fillSavedList();
      this.$("savedNet").value = "";
      this.$("watchParent").checked = true;
      this.$("watchGeneration").checked = false;
      this.$("watchEver").checked = false;
      this.resetEpisode();
      this.logEvent("JSON load succeeded");
    };
    this.$("randomWeights").onclick = () => {
      const keptGen = this.state.gen;
      const keptHist = this.state.hist;
      this.state = new EvolutionSearch(this.randomNormal);
      for (let i = 0; i < this.state.theta.length; i++) this.state.theta[i] = (this.random() < 0.5 ? -1 : 1) * (config.search.randomWeightAbsMin + this.random() * (config.search.randomWeightAbsMax - config.search.randomWeightAbsMin));
      this.state.bestGenTheta = this.state.theta.slice();
      this.state.bestEverTheta = this.state.theta.slice();
      this.state.gen = keptGen;
      this.state.hist = keptHist;
      this.state.snapshots = [];
      this.backedBestScore = this.state.bestEverScore ?? -1e9;
      this.visit.parentName = "";
      this.visit.loadedLabel = "";
      this.visit.trained = false;
      this.fillSavedList();
      this.saveSession();
      this.resetEpisode();
      this.logEvent("random weights");
    };
    this.$("log").onclick = () => {
      this.paintLog();
      this.toggleDialog(this.$("logDialog"));
      if (this.$("logDialog").open) this.$("logBody").scrollTop = this.$("logBody").scrollHeight;
    };
    this.$("logClose").onclick = () => this.$("logDialog").close();
    this.$("graphBtn").onclick = () => {
      const dialog = this.$("graphDialog");
      if (dialog.open) dialog.close();
      else {
        dialog.show();
        requestAnimationFrame(() => { this.drawNetworkGraph(); this.paintNetList(); });
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
    };
    this.$("speed").oninput = () => { this.$("speedValue").textContent = this.$("speed").value + " m/s"; };

    this.app = new PIXI.Application();
    this.host = this.$("stage");
    await this.app.init({ resizeTo: this.host, background: 0x12162a, antialias: true, resolution: window.devicePixelRatio || 1, autoDensity: true });
    this.host.appendChild(this.app.canvas);
    this.graphics = new PIXI.Graphics();
    this.throws = new PIXI.Graphics();
    this.chart = new PIXI.Graphics();
    this.doll = new PIXI.Container();
    this.doll.sortableChildren = true;
    this.label = new PIXI.Text({ text: "", style: { fill: 0xdfe4ff, fontSize: 13, fontFamily: "system-ui,sans-serif", lineHeight: 18 } });
    this.label.position.set(12, 10);
    this.app.stage.addChild(this.graphics, this.doll, this.throws, this.chart, this.label);
    this.resetEpisode();
    this.$("trainPhase").onchange = () => this.applyTrainPhase();
    this.applyTrainPhase();
    this.trainLoop();

    this.canvas = this.app.canvas;
    this.canvas.style.touchAction = "none";
    this.colors = [0x4f8cff, 0xffd166, 0x7aa2ff, 0x7aa2ff, 0xffd166, 0x43d9ad, 0x43d9ad, 0xf78c6c];
    this.limbColor = (index) => this.colors[2 + (index - 2) % 6];
    this.pantsColor = this.limbColor(config.partIndex.thighLeft);
    this.darken = (color) => {
      const scale = 0.62;
      const red = (color >> 16) & 255;
      const green = (color >> 8) & 255;
      const blue = color & 255;
      return (Math.round(red * scale) << 16) | (Math.round(green * scale) << 8) | Math.round(blue * scale);
    };
    this.partColor = (index) => {
      if (index < 2) return this.colors[index];
      const color = this.limbColor(index);
      return index < config.partIndex.upperArmRight ? this.darken(color) : color;
    };
    this.solidTexture = (circle) => {
      const size = 32;
      const textureCanvas = document.createElement("canvas");
      textureCanvas.width = size;
      textureCanvas.height = size;
      const context = textureCanvas.getContext("2d");
      context.fillStyle = "#ffffff";
      if (circle) {
        context.beginPath();
        context.arc(size / 2, size / 2, size / 2, 0, Math.PI * 2);
        context.fill();
      } else context.fillRect(0, 0, size, size);
      return PIXI.Texture.from(textureCanvas);
    };
    this.boxTexture = this.solidTexture(false);
    this.circleTexture = this.solidTexture(true);
    this.depth = 1;
    this.addDollSprite = (texture, tint) => {
      const sprite = new PIXI.Sprite(texture);
      sprite.anchor.set(0.5);
      sprite.tint = tint;
      sprite.zIndex = this.depth++;
      this.doll.addChild(sprite);
      return sprite;
    };
    this.partSprites = [];
    this.addPartSprite = (index) => {
      this.partSprites[index] = this.addDollSprite(this.boxTexture, this.partColor(index));
    };
    for (const index of [config.partIndex.thighLeft, config.partIndex.shinLeft, config.partIndex.footLeft]) this.addPartSprite(index);
    this.leftKnee = this.world.joints.find((joint) => joint.mark === "knee" && joint.a === config.partIndex.thighLeft);
    this.leftKneeSprite = this.addDollSprite(this.circleTexture, this.partColor(config.partIndex.thighLeft));
    for (const index of [config.partIndex.upperArmLeft, config.partIndex.forearmLeft, config.partIndex.handLeft]) this.addPartSprite(index);
    this.tailSprite = this.addDollSprite(this.circleTexture, this.pantsColor);
    this.hipMarks = this.world.joints.filter((joint) => joint.mark === "hip");
    for (const index of [config.partIndex.torso, config.partIndex.head]) this.addPartSprite(index);
    for (const index of [config.partIndex.thighRight, config.partIndex.shinRight, config.partIndex.footRight]) this.addPartSprite(index);
    this.rightKnee = this.world.joints.find((joint) => joint.mark === "knee" && joint.a === config.partIndex.thighRight);
    this.rightKneeSprite = this.addDollSprite(this.circleTexture, this.partColor(config.partIndex.thighRight));
    for (const index of [config.partIndex.upperArmRight, config.partIndex.forearmRight, config.partIndex.handRight]) this.addPartSprite(index);
    this.accumulator = 0;
    this.frameSkip = 0;
    this.canvas.addEventListener("pointerdown", (event) => {
      const screen = this.screenPoint(event);
      const scale = this.view.scale || 100;
      const groundY = this.view.groundY || this.app.screen.height * 0.78;
      const point = this.worldPoint(screen.x, screen.y, scale, groundY);
      this.canvas.setPointerCapture(event.pointerId);
      let picked = null;
      let best = 1e9;
      for (const part of this.world.parts) {
        const position = b2Body_GetPosition(part.id);
        const distance = (position.x - point.x) ** 2 + (position.y - point.y) ** 2;
        if (distance < best && distance < (part.hx + 0.2) ** 2) { best = distance; picked = part; }
      }
      if (picked) {
        const joint = CreateMouseJoint({ worldId: this.world.worldId, bodyIdA: this.world.groundId, bodyIdB: picked.id, target: this.vector(point.x, point.y), hertz: 5, dampingRatio: 0.7, maxForce: 1500 });
        this.drag = { id: joint.jointId, x: screen.x, y: screen.y, part: picked };
      } else this.aim = { x0: screen.x, y0: screen.y, x: screen.x, y: screen.y };
    });
    this.canvas.addEventListener("pointermove", (event) => {
      const screen = this.screenPoint(event);
      if (this.drag) { this.drag.x = screen.x; this.drag.y = screen.y; }
      if (this.aim) { this.aim.x = screen.x; this.aim.y = screen.y; }
    });
    this.canvas.addEventListener("pointerup", () => this.release());
    this.canvas.addEventListener("pointercancel", () => this.release());
    this.$("fire").onclick = () => {
      const side = Math.random() < 0.5 ? -1 : 1;
      const speed = +this.$("speed").value;
      const kind = thrownKinds[+this.$("objectKind").value];
      const id = this.sim.spawnThrown(this.world, kind, this.episode.measure.centerX + side * 3.5, 1.2, -side * speed, 1.5);
      this.objects.push({ id, kind });
    };

    this.app.ticker.add((ticker) => {
      this.accumulator += Math.min(ticker.deltaMS, 50);
      while (this.accumulator >= 1000 / config.physics.hz) {
        this.accumulator -= 1000 / config.physics.hz;
        if (this.hold > 0) { this.hold--; if (!this.hold) this.resetEpisode(); }
        else if (this.episode.step()) this.hold = 60;
      }
      const width = this.app.screen.width;
      const height = this.app.screen.height;
      const scale = Math.min(height / 2.8, 150);
      const groundY = height * 0.78;
      this.view.scale = scale;
      this.view.groundY = groundY;
      this.view.cx += ((this.episode.measure.centerX || 0) - this.view.cx) * Math.min(1, ticker.deltaMS / 250);
      if (this.drag) {
        const point = this.worldPoint(this.drag.x, this.drag.y, scale, groundY);
        b2MouseJoint_SetTarget(this.drag.id, this.vector(point.x, point.y));
      }
      const toX = (x) => (x - this.view.cx) * scale + width / 2;
      const toY = (y) => groundY - y * scale;
      this.graphics.clear().rect(0, toY(-0.5), width, height).fill(0x232a47);
      for (let mark = Math.floor(this.view.cx - width / scale / 2) - 1; mark <= this.view.cx + width / scale / 2 + 1; mark++) {
        this.graphics.rect(toX(mark) - 1, toY(-0.5), 2, 8).fill(0x3a4570);
      }
      const platform = b2Body_GetPosition(this.world.platformId);
      const platformAngle = b2Rot_GetAngle(b2Body_GetRotation(this.world.platformId));
      const platformCos = Math.cos(platformAngle);
      const platformSin = Math.sin(platformAngle);
      const corners = [];
      for (const [signX, signY] of [[-1, -1], [1, -1], [1, 1], [-1, 1]]) {
        const localX = signX * config.body.platformHalfWidth;
        const localY = signY * config.body.platformHalfThickness;
        corners.push(toX(platform.x + platformCos * localX - platformSin * localY), toY(platform.y + platformSin * localX + platformCos * localY));
      }
      this.graphics.poly(corners).fill(0x5a6aa0);
      if (this.episode.options.goalX != null) {
        this.graphics.circle(toX(this.episode.options.goalX), toY(0.08), 9).fill(0x43d9ad);
        this.graphics.moveTo(toX(this.episode.options.goalX), toY(0)).lineTo(toX(this.episode.options.goalX), toY(1.2)).stroke({ width: 3, color: 0x43d9ad });
        this.graphics.circle(toX(this.episode.options.goalX), toY(1.2), 7).fill(0x43d9ad);
      }
      const placePart = (index) => {
        const part = this.world.parts[index];
        const sprite = this.partSprites[index];
        const position = b2Body_GetPosition(part.id);
        const x = position.x;
        const y = position.y;
        const angle = b2Rot_GetAngle(b2Body_GetRotation(part.id));
        sprite.position.set(toX(x), toY(y));
        sprite.rotation = -angle;
        sprite.width = part.hx * 2 * scale;
        sprite.height = part.hy * 2 * scale;
      };
      const anchorOf = (joint) => {
        const part = this.world.parts[joint.a];
        const position = b2Body_GetPosition(part.id);
        const x = position.x;
        const y = position.y;
        const angle = b2Rot_GetAngle(b2Body_GetRotation(part.id));
        const cosine = Math.cos(angle);
        const sine = Math.sin(angle);
        return { x: x + cosine * joint.ax - sine * joint.ay, y: y + sine * joint.ax + cosine * joint.ay };
      };
      const placeKnee = (joint, sprite) => {
        const point = anchorOf(joint);
        sprite.position.set(toX(point.x), toY(point.y));
        sprite.width = 0.11 * scale;
        sprite.height = 0.11 * scale;
      };
      for (const index of [config.partIndex.thighLeft, config.partIndex.shinLeft, config.partIndex.footLeft, config.partIndex.upperArmLeft, config.partIndex.forearmLeft, config.partIndex.handLeft, config.partIndex.torso, config.partIndex.head, config.partIndex.thighRight, config.partIndex.shinRight, config.partIndex.footRight, config.partIndex.upperArmRight, config.partIndex.forearmRight, config.partIndex.handRight]) placePart(index);
      placeKnee(this.leftKnee, this.leftKneeSprite);
      placeKnee(this.rightKnee, this.rightKneeSprite);
      if (this.hipMarks.length === 2) {
        let hipX = 0;
        let hipY = 0;
        for (const joint of this.hipMarks) {
          const point = anchorOf(joint);
          hipX += point.x;
          hipY += point.y;
        }
        const torso = this.world.parts[config.partIndex.torso];
        const torsoAngle = b2Rot_GetAngle(b2Body_GetRotation(torso.id));
        const frontX = Math.cos(torsoAngle);
        const frontY = Math.sin(torsoAngle);
        const tailRadius = torso.hx * 1.1;
        this.tailSprite.position.set(toX(hipX / 2 + frontX * (torso.hx - tailRadius)), toY(hipY / 2 + frontY * (torso.hx - tailRadius)));
        this.tailSprite.width = tailRadius * 2 * scale;
        this.tailSprite.height = tailRadius * 2 * scale;
      }
      this.throws.clear();
      for (const thrown of this.objects) {
        const position = b2Body_GetPosition(thrown.id);
        if (thrown.kind.box) {
          const angle = b2Rot_GetAngle(b2Body_GetRotation(thrown.id));
          const cosine = Math.cos(angle);
          const sine = Math.sin(angle);
          const points = [];
          for (const [signX, signY] of [[-1, -1], [1, -1], [1, 1], [-1, 1]]) {
            points.push(toX(position.x + cosine * signX * thrown.kind.half - sine * signY * thrown.kind.half), toY(position.y + sine * signX * thrown.kind.half + cosine * signY * thrown.kind.half));
          }
          this.throws.poly(points).fill(thrown.kind.color);
        } else this.throws.circle(toX(position.x), toY(position.y), thrown.kind.radius * scale).fill(thrown.kind.color);
      }
      if (++this.frameSkip % 8 === 0) {
        const scored = scoreDecision(this.episode.measure);
        this.paintScore(scored);
        if (this.$("graphDialog").open) {
          this.drawNetworkGraph();
          this.paintNetList();
        }
        const last = this.state.hist[this.state.hist.length - 1];
        const pose = isStanding(this.episode.measure) ? "standing" : (this.episode.measure.handOn ? "on hands" : (this.episode.measure.bodyOn ? "on the floor" : "in the air"));
        const goal = this.episode.options.goalX == null ? "" : " · goal " + Math.abs(this.episode.options.goalX - this.episode.measure.centerX).toFixed(1) + " m";
        const fitness = last ? " · mean " + last.mean.toFixed(2) + " best " + last.best.toFixed(2) : "";
        this.$("status").textContent = (this.paused ? "idle" : "running") + " · " + this.trainPhaseLabel() + " · watching " + this.watchModeLabel() + " · " + this.networkTitle() + " · gen " + this.state.gen + fitness;
        this.label.text = "Head " + (this.episode.measure.headHeight || 0).toFixed(2) + " m · torso " + (Math.abs(this.episode.measure.torsoAngle) * 180 / Math.PI).toFixed(0) + "° · " + pose + goal;
      }
      this.drawChart();
    });
  }

  bootWorkers() {
    const count = Math.max(config.workers.browserMin, Math.min(config.workers.browserMax, (navigator.hardwareConcurrency || config.workers.browserHardwareFallback) - config.workers.browserReserve));
    return Promise.all(Array.from({ length: count }, () => new Promise((resolve, reject) => {
      const worker = new Worker("./src/browser-worker.js", { type: "module" });
      const timer = setTimeout(() => reject(new Error("A worker did not answer in 15 s")), 15000);
      worker.onmessage = (event) => { if (event.data === "ready") { clearTimeout(timer); resolve(); } };
      worker.onerror = () => { clearTimeout(timer); reject(new Error("Could not start a worker")); };
      this.pool.workers.push(worker);
    }))).then(() => { this.pool.ready = true; });
  }

  evaluatePopulation(items) {
    const buckets = this.pool.workers.map(() => []);
    items.forEach((item, index) => buckets[index % this.pool.workers.length].push(item));
    return Promise.all(buckets.map((bucket, workerIndex) => {
      if (!bucket.length) return [];
      return new Promise((resolve, reject) => {
        const worker = this.pool.workers[workerIndex];
        worker.onmessage = (event) => resolve(event.data);
        worker.onerror = () => reject(new Error("Worker failed"));
        worker.postMessage({ tasks: bucket, trainingPhase: config.curriculum.phase });
      });
    })).then((groups) => {
      const flat = groups.flat();
      const ordered = new Array(items.length);
      for (const row of flat) ordered[row.index] = row;
      return ordered;
    });
  }

  async trainLoop() {
    try { await this.bootWorkers(); }
    catch (error) { this.$("error").textContent = error.message; return; }
    while (this.pool.ready) {
      if (this.paused) { await new Promise((resolve) => setTimeout(resolve, 150)); continue; }
      const generationState = this.state;
      try {
        const generation = generationState.ask(this.sim.situation(this.random, this.$("trainPose").value), Math.max(config.workers.browserPairMin, this.pool.workers.length), this.randomNormal);
        const starts = [];
        for (let pair = 0; pair < generation.count; pair++) starts.push(this.sim.varyStart(generation.situation, this.random));
        const tasks = generation.weights.map((weights, index) => ({
          index,
          weights,
          phase: index % config.cadence.actionEveryNFrames,
          options: starts[Math.floor(index / 2)] || generation.situation,
        }));
        const scores = await this.evaluatePopulation(tasks);
        if (this.state !== generationState) continue;
        if (scores.some((score) => !score || !Number.isFinite(score.fitness))) throw new Error("A worker returned an empty score");
        this.state.tell(generation, scores);
        await this.rememberBest();
        this.saveSession();
        this.visit.trained = true;
        const finished = this.state.hist[this.state.hist.length - 1];
        if (finished) this.logEvent("generation finished (gen " + this.state.gen + ", mean " + finished.mean.toFixed(2) + ", best " + finished.best.toFixed(2) + ")");
        this.advanceTrainPhase();
      } catch (error) {
        this.$("error").textContent = error.message;
        this.pool.ready = false;
      }
    }
  }

  savedSnapshot() {
    const generation = this.$("savedNet").value;
    if (!generation) return null;
    return (this.state.snapshots || []).find((item) => String(item.gen) === generation) || null;
  }

  watchedWeights() {
    const saved = this.savedSnapshot();
    if (saved) return saved.weights;
    if (this.$("watchEver").checked) return this.state.bestEverTheta || this.state.theta;
    if (this.$("watchGeneration").checked) return this.state.bestGenTheta || this.state.theta;
    return this.state.theta;
  }

  watchModeLabel() {
    if (this.savedSnapshot()) return "saved";
    if (this.$("watchEver").checked) return "best ever";
    if (this.$("watchGeneration").checked) return "best this generation";
    return "parent";
  }

  trainPhaseLabel() {
    const name = activeTrainingPhase().name;
    return config.curriculum.byPhases ? "By phases: " + name : name;
  }

  applyTrainPhase() {
    const value = this.$("trainPhase").value;
    if (value === "phases") {
      if (!config.curriculum.byPhases) {
        setTrainingByPhases(true);
        setTrainingPhase(config.curriculum.phases[0].id);
        this.phaseStartGen = this.state.gen || 0;
      }
    } else {
      setTrainingByPhases(false);
      setTrainingPhase(+value);
    }
  }

  advanceTrainPhase() {
    if (!config.curriculum.byPhases) return;
    const phase = activeTrainingPhase();
    const index = config.curriculum.phases.findIndex((item) => item.id === phase.id);
    if (index < 0 || index >= config.curriculum.phases.length - 1) return;
    if ((this.state.gen || 0) - this.phaseStartGen < phase.generations) return;
    const next = config.curriculum.phases[index + 1];
    setTrainingPhase(next.id);
    this.phaseStartGen = this.state.gen || 0;
    this.logEvent("By phases: " + next.name + " (gen " + this.state.gen + ")");
  }

  networkTitle() {
    const saved = this.savedSnapshot();
    if (saved) return saved.name || ("gen " + saved.gen);
    if (this.$("watchEver").checked) {
      const gen = Number.isFinite(this.state.bestEverGen) ? this.state.bestEverGen : this.state.gen;
      return "Best ever, generation " + gen + (this.state.bestEverName ? ", " + this.state.bestEverName : "");
    }
    if (this.$("watchGeneration").checked) {
      return "Best this generation, generation " + this.state.gen + (this.state.bestGenName ? ", " + this.state.bestGenName : "");
    }
    if (this.visit.parentName) return this.visit.parentName;
    if (this.visit.loadedLabel && this.paused && !this.visit.trained) return "Loaded " + this.visit.loadedLabel;
    return "Parent, generation " + this.state.gen;
  }

  logEvent(line) {
    const now = new Date();
    const pad = (n) => String(n).padStart(2, "0");
    this.visit.log.push(pad(now.getHours()) + ":" + pad(now.getMinutes()) + ":" + pad(now.getSeconds()) + "  " + line);
    const body = this.$("logBody");
    body.textContent = this.visit.log.join("\n");
    body.scrollTop = body.scrollHeight;
  }

  paintLog() {
    this.$("logBody").textContent = this.visit.log.length ? this.visit.log.join("\n") : "Nothing yet.";
  }

  toggleDialog(dialog) {
    if (dialog.open) dialog.close();
    else dialog.show();
  }

  episodeOptions() {
    const pose = this.$("startPose").value;
    const side = Math.random() < 0.5 ? -1 : 1;
    const goalX = side * (1.2 + Math.random());
    if (pose === "up") return { tilt: (Math.random() * 2 - 1) * 0.1, drop: 0.12, settle: 0, frames: 1e9, goalX, goalY: 0, amp: 0, period: 8, phaseShift: 0 };
    const sign = pose === "prone" ? -1 : 1;
    return { tilt: sign * (1.4 + Math.random() * 0.2), drop: 0.04, settle: 40, frames: 1e9, goalX, goalY: 0, amp: 0, period: 8, phaseShift: 0 };
  }

  resetEpisode() {
    if (this.episode) this.episode.cleanup();
    const options = this.episodeOptions();
    this.episode = new this.sim.Episode(this.world, this.watchedWeights(), options.tilt, options.drop, options);
    this.episode.free = true;
    this.hold = 0;
    if (this.$("graphDialog").open) {
      this.drawNetworkGraph();
      this.paintNetList();
    }
  }

  paintScore(scored) {
    const total = this.$("total");
    total.className = scored.total >= 0 ? "good" : "bad";
    total.innerHTML = scored.total.toFixed(2) + '<small title="The ceiling if every row is at its best.">max ' + scored.max.toFixed(2) + "</small>";
    this.$("terms").innerHTML = scored.terms.map((term) => {
      const width = Math.min(48, (Math.abs(term.v) / 1.2) * 48);
      const left = term.v >= 0 ? 50 : 50 - width;
      const color = term.v >= 0 ? "#3ecf8e" : "#ff4d5a";
      return '<div class="row" title="' + this.escapeTip(term.tip) + '"><span>' + term.n + '</span><span class="bar"><i style="left:' + left + '%;width:' + width + '%;background:' + color + '"></i></span><span class="' + (term.v >= 0 ? "good" : "bad") + '">' + (term.v >= 0 ? "+" : "") + term.v.toFixed(2) + "</span></div>";
    }).join("");
  }

  paintNetList() {
    const input = this.episode && this.episode.input;
    const hidden = this.episode && this.episode.hidden;
    const action = this.episode && this.episode.action;
    if (!input || !hidden || !action) { this.$("netlist").innerHTML = ""; return; }
    const line = (name, value, tip) => '<div class="row" title="' + this.escapeTip(tip) + '"><span>' + name + '</span><span class="bar"><i style="width:' + Math.min(100, Math.abs(value) * 100) + '%;background:' + (value >= 0 ? "#3ecf8e" : "#ff4d5a") + '"></i></span><span>' + (value >= 0 ? "+" : "") + value.toFixed(2) + "</span></div>";
    let html = "<h3>Inputs (" + input.length + ")</h3>";
    for (let i = 0; i < input.length; i++) {
      const rowLabel = this.observationLabel(i);
      html += line(rowLabel.name, input[i], rowLabel.tip);
    }
    html += "<h3>Hidden (" + hidden.length + ")</h3>";
    for (let unit = 0; unit < hidden.length; unit++) html += line(String(unit), hidden[unit], "Hidden unit " + unit + ". It keeps a fraction of its previous value.");
    html += "<h3>Outputs</h3>";
    for (let joint = 0; joint < action.length; joint++) html += line(jointNames[joint], action[joint], "Command to " + jointNames[joint] + ". Positive and negative bend opposite ways.");
    this.$("netlist").innerHTML = html;
  }

  observationLabel(index) {
    return observationLabels[index] || { name: "input " + index, tip: "Observation " + index + "." };
  }

  escapeTip(text) { return String(text).replace(/"/g, "&quot;"); }

  paintPause() {
    this.$("pause").textContent = this.paused ? "Start training" : "Pause";
    this.$("pause").title = this.paused
      ? "Loading this page does not train until you press this. The doll on screen keeps moving."
      : "Stops the search. The doll on screen keeps moving.";
  }

  screenPoint(event) {
    const rect = this.canvas.getBoundingClientRect();
    return { x: event.clientX - rect.left, y: event.clientY - rect.top };
  }

  worldPoint(screenX, screenY, scale, groundY) {
    return { x: (screenX - this.app.screen.width / 2) / scale + this.view.cx, y: (groundY - screenY) / scale };
  }

  release() {
    if (this.drag) { b2DestroyJoint(this.drag.id); this.drag = null; }
    if (this.aim) {
      const scale = this.view.scale || 100;
      const groundY = this.view.groundY || this.app.screen.height * 0.78;
      const start = this.worldPoint(this.aim.x0, this.aim.y0, scale, groundY);
      let vx = (this.aim.x - this.aim.x0) / scale * 5;
      let vy = -(this.aim.y - this.aim.y0) / scale * 5;
      const speed = Math.hypot(vx, vy);
      if (speed > 25) { vx *= 25 / speed; vy *= 25 / speed; }
      const kind = thrownKinds[+this.$("objectKind").value];
      const id = this.sim.spawnThrown(this.world, kind, start.x, Math.max(start.y, 0.2), vx, vy);
      this.objects.push({ id, kind });
      if (this.objects.length > 15) b2DestroyBody(this.objects.shift().id);
      this.aim = null;
    }
  }

  drawChart() {
    this.chart.clear();
    const width = Math.min(240, this.app.screen.width - 24);
    const height = 70;
    const originX = this.app.screen.width - width - 12;
    this.chart.roundRect(originX, 12, width, height, 6).fill({ color: 0x000000, alpha: 0.35 });
    const history = this.state.hist;
    if (history.length < 2) return;
    for (const [key, color] of [["mean", 0x6f95ff], ["best", 0x43d9ad]]) {
      history.forEach((sample, index) => {
        const x = originX + 4 + index / (history.length - 1) * (width - 8);
        const y = 12 + height - 4 - clamp((sample[key] + 2) / 10, 0, 1) * (height - 8);
        if (index) this.chart.lineTo(x, y);
        else this.chart.moveTo(x, y);
      });
      this.chart.stroke({ width: 1.5, color });
    }
  }

drawNetworkGraph() {
    const canvas = this.$("graphCanvas");
    const missing = this.$("graphMissing");
    this.$("graphTitle").textContent = this.networkTitle();
    const episode = this.episode;
    const weights = episode && episode.weights;
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
    const inputs = episode.input;
    const firstAct = episode.network && episode.network.compressed;
    const memoryAct = episode.hidden;
    const outputs = episode.action;
    const joints = jointNames.length;
    const width = Math.max(320, canvas.clientWidth || 640);
    const height = Math.max(360, canvas.clientHeight || 480);
    const pixelRatio = Math.min(1.5, devicePixelRatio || 1);
    if (canvas.width !== Math.floor(width * pixelRatio) || canvas.height !== Math.floor(height * pixelRatio)) {
      canvas.width = Math.floor(width * pixelRatio);
      canvas.height = Math.floor(height * pixelRatio);
    }
    context.setTransform(pixelRatio, 0, 0, pixelRatio, 0, 0);
    context.clearRect(0, 0, width, height);
    context.font = "11px system-ui, sans-serif";
    context.textBaseline = "middle";
    if (!this.drawNetworkGraph.labelPad) {
      let widest = 72;
      for (let i = 0; i < config.net.inputCount; i++) widest = Math.max(widest, context.measureText(this.observationLabel(i).name).width);
      this.drawNetworkGraph.labelPad = Math.ceil(widest + 16);
    }
    const left = Math.min(this.drawNetworkGraph.labelPad, width * 0.32);
    const right = width - 88;
    const top = 28;
    const bottom = height - 12;
    const yOf = (index, count) => count <= 1 ? (top + bottom) / 2 : top + index * (bottom - top) / (count - 1);
    const columns = feedforward
      ? [
          { title: "", count: config.net.inputCount, x: left, radius: 1.6, values: inputs, tag: (i) => this.observationLabel(i).name, align: "right" },
          { title: "hidden 1", count: feedforward, x: left + (right - left) * 0.38, radius: 3.5, values: firstAct, tag: (i) => "a" + i, align: "left" },
          { title: "hidden 2", count: recurrent, x: left + (right - left) * 0.68, radius: 3.5, values: memoryAct, tag: (i) => "b" + i, align: "left" },
          { title: "", count: joints, x: right, radius: 5, values: outputs, tag: (i) => jointNames[i], align: "left" },
        ]
      : [
          { title: "", count: config.net.inputCount, x: left, radius: 1.6, values: inputs, tag: (i) => this.observationLabel(i).name, align: "right" },
          { title: "recurrent dashed", count: recurrent, x: Math.round(width * 0.58), radius: 5, values: memoryAct, tag: (i) => "h" + i, align: "left" },
          { title: "", count: joints, x: right, radius: 5, values: outputs, tag: (i) => jointNames[i], align: "left" },
        ];
    const yAt = (column, index) => yOf(index, column.count);
    const bucketCount = 6;
    const alphas = [0.06, 0.11, 0.18, 0.3, 0.46, 0.7];
    const positiveFeed = [];
    const negativeFeed = [];
    const positiveMemory = [];
    const negativeMemory = [];
    for (let bucket = 0; bucket < bucketCount; bucket++) {
      positiveFeed.push(new Path2D());
      negativeFeed.push(new Path2D());
      positiveMemory.push(new Path2D());
      negativeMemory.push(new Path2D());
    }
    const bucketOf = (weight, source) => {
      const drive = Math.min(1, Math.abs(source || 0) * Math.min(1, Math.abs(weight)));
      return Math.min(bucketCount - 1, (drive * bucketCount) | 0);
    };
    const addLine = (path, x0, y0, x1, y1) => { path.moveTo(x0, y0); path.lineTo(x1, y1); };
    const wire = (weight, source, x0, y0, x1, y1) => {
      addLine((weight >= 0 ? positiveFeed : negativeFeed)[bucketOf(weight, source)], x0, y0, x1, y1);
    };
    const memoryArc = (weight, source, x, y0, y1, self) => {
      const path = (weight >= 0 ? positiveMemory : negativeMemory)[bucketOf(weight, source)];
      if (self) {
        const start = 0.7;
        path.moveTo(x - 14 + Math.cos(start) * 6, y1 + Math.sin(start) * 6);
        path.arc(x - 14, y1, 6, start, Math.PI * 2.15);
      } else {
        path.moveTo(x, y0);
        path.quadraticCurveTo(x - 36, (y0 + y1) / 2, x, y1);
      }
    };
    const inputColumn = columns[0];
    const jointColumn = columns[columns.length - 1];
    if (!feedforward) {
      const hiddenColumn = columns[1];
      const stride = 1 + config.net.inputCount + recurrent;
      const outStride = recurrent + 1;
      const outBase = recurrent * stride;
      for (let unit = 0; unit < recurrent; unit++) {
        const base = unit * stride;
        const hiddenY = yAt(hiddenColumn, unit);
        for (let i = 0; i < config.net.inputCount; i++) {
          wire(weights[base + 1 + i], inputs && inputs[i], inputColumn.x, yAt(inputColumn, i), hiddenColumn.x, hiddenY);
        }
        for (let other = 0; other < recurrent; other++) {
          memoryArc(weights[base + 1 + config.net.inputCount + other], memoryAct && memoryAct[other], hiddenColumn.x, yAt(hiddenColumn, other), hiddenY, other === unit);
        }
      }
      for (let joint = 0; joint < joints; joint++) {
        for (let unit = 0; unit < recurrent; unit++) {
          wire(weights[outBase + joint * outStride + 1 + unit], memoryAct && memoryAct[unit], hiddenColumn.x, yAt(hiddenColumn, unit), jointColumn.x, yAt(jointColumn, joint));
        }
      }
    } else {
      const firstColumn = columns[1];
      const memoryColumn = columns[2];
      for (let unit = 0; unit < feedforward; unit++) {
        const base = unit * layout.feedforwardStride;
        const hiddenY = yAt(firstColumn, unit);
        for (let i = 0; i < config.net.inputCount; i++) {
          wire(weights[base + 1 + i], inputs && inputs[i], inputColumn.x, yAt(inputColumn, i), firstColumn.x, hiddenY);
        }
      }
      for (let unit = 0; unit < recurrent; unit++) {
        const base = layout.feedforwardBlock + unit * layout.recurrentStride;
        const hiddenY = yAt(memoryColumn, unit);
        for (let i = 0; i < feedforward; i++) {
          wire(weights[base + 1 + i], firstAct && firstAct[i], firstColumn.x, yAt(firstColumn, i), memoryColumn.x, hiddenY);
        }
        for (let other = 0; other < recurrent; other++) {
          memoryArc(weights[base + 1 + feedforward + other], memoryAct && memoryAct[other], memoryColumn.x, yAt(memoryColumn, other), hiddenY, other === unit);
        }
      }
      for (let joint = 0; joint < joints; joint++) {
        const base = layout.outputBase + joint * (recurrent + 1);
        for (let unit = 0; unit < recurrent; unit++) {
          wire(weights[base + 1 + unit], memoryAct && memoryAct[unit], memoryColumn.x, yAt(memoryColumn, unit), jointColumn.x, yAt(jointColumn, joint));
        }
      }
    }
    context.lineWidth = 0.55;
    context.setLineDash([]);
    for (let bucket = 0; bucket < bucketCount; bucket++) {
      context.strokeStyle = "rgba(62,207,142," + alphas[bucket] + ")";
      context.stroke(positiveFeed[bucket]);
      context.strokeStyle = "rgba(255,77,90," + alphas[bucket] + ")";
      context.stroke(negativeFeed[bucket]);
    }
    context.lineWidth = 1;
    context.setLineDash([4, 3]);
    for (let bucket = 0; bucket < bucketCount; bucket++) {
      context.strokeStyle = "rgba(62,207,142," + alphas[bucket] + ")";
      context.stroke(positiveMemory[bucket]);
      context.strokeStyle = "rgba(255,77,90," + alphas[bucket] + ")";
      context.stroke(negativeMemory[bucket]);
    }
    context.setLineDash([]);
    const fillOf = (value) => {
      const mag = Math.min(1, Math.abs(value || 0));
      const alpha = 0.18 + 0.82 * mag;
      return (value || 0) >= 0 ? "rgba(62,207,142," + alpha + ")" : "rgba(255,77,90," + alpha + ")";
    };
    const dot = (x, y, radius, value) => {
      context.fillStyle = fillOf(value);
      context.beginPath();
      context.arc(x, y, radius, 0, Math.PI * 2);
      context.fill();
    };
    for (const column of columns) {
      for (let index = 0; index < column.count; index++) dot(column.x, yAt(column, index), column.radius, column.values && column.values[index]);
    }
    const tag = (text, x, y, align) => {
      context.textAlign = align;
      const box = context.measureText(text).width;
      const boxX = align === "right" ? x - box : align === "center" ? x - box / 2 : x;
      context.fillStyle = "#12162a";
      context.fillRect(boxX - 2, y - 7, box + 4, 14);
      context.fillStyle = "#e7e9f5";
      context.fillText(text, x, y);
    };
    for (const column of columns) if (column.title) tag(column.title, column.x, 11, "center");
    const labeled = [];
    const usedY = [];
    const place = (index) => {
      const y = yAt(inputColumn, index);
      for (let k = 0; k < usedY.length; k++) if (Math.abs(usedY[k] - y) < 12) return;
      usedY.push(y);
      labeled.push(index);
    };
    const labelStep = Math.max(4, Math.ceil(config.net.inputCount / Math.max(6, ((bottom - top) / 15) | 0)));
    for (let i = 0; i < config.net.inputCount; i += labelStep) place(i);
    place(config.net.inputCount - 1);
    if (inputs && inputs.length) {
      const rank = [];
      for (let i = 0; i < config.net.inputCount; i++) rank.push(i);
      rank.sort((a, b) => Math.abs(inputs[b] || 0) - Math.abs(inputs[a] || 0));
      for (let k = 0; k < 8 && k < rank.length; k++) place(rank[k]);
    }
    for (const index of labeled) tag(inputColumn.tag(index), inputColumn.x - 8, yAt(inputColumn, index), "right");
    for (const column of columns.slice(1)) {
      const step = Math.max(1, Math.ceil(column.count / Math.max(6, (bottom - top) / 15)));
      const seen = [];
      const keep = (index) => {
        const y = yAt(column, index);
        for (let k = 0; k < seen.length; k++) if (Math.abs(seen[k] - y) < 12) return;
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
    return date.getFullYear() + pad(date.getMonth() + 1) + pad(date.getDate()) + "-" + pad(date.getHours()) + pad(date.getMinutes()) + pad(date.getSeconds());
  }

  async thetaHash8(weights) {
    const packed = JSON.stringify(Array.from(weights, (value) => +(+value).toFixed(4)));
    const digest = await crypto.subtle.digest("SHA-256", new TextEncoder().encode(packed));
    return Array.from(new Uint8Array(digest), (byte) => byte.toString(16).padStart(2, "0")).join("").slice(0, 8);
  }

  async freeStem(hash) {
    let when = new Date();
    for (let n = 0; n < 5; n++) {
      const stem = this.fileStamp(when) + "_" + hash;
      try { await this.backupDir.getFileHandle(stem + ".json"); }
      catch (error) { return stem; }
      when = new Date(when.getTime() + 1000);
    }
    return this.fileStamp(new Date()) + "_" + hash;
  }

  backupPayload(role, stem) {
    const pack = (weights) => weights ? Array.from(weights, (value) => +value.toFixed(4)) : null;
    return {
      role,
      name: stem,
      gen: this.state.gen,
      bestEverScore: this.state.bestEverScore,
      bestEverGen: this.state.bestEverGen || 0,
      bestEverName: role === "best-ever" ? stem : (this.state.bestEverName || ""),
      trainingPhase: config.curriculum.phase,
      inputCount: config.net.inputCount,
      theta: pack(this.state.theta),
      bestEverTheta: pack(this.state.bestEverTheta),
    };
  }

  async writeBackup(role) {
    const weights = role === "best-ever" ? this.state.bestEverTheta : this.state.theta;
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
    } catch (error) { return ""; }
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
    this.state.snapshots.push({
      gen: this.state.gen,
      score: this.state.bestEverScore,
      weights: Float32Array.from(this.state.bestEverTheta),
      name: this.state.bestEverName || "",
    });
    if (this.state.snapshots.length > 40) this.state.snapshots.shift();
    this.fillSavedList();
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
      option.textContent = "gen " + item.gen + " · " + Number(item.score).toFixed(2);
      select.appendChild(option);
    }
    const keep = [...select.options].some((option) => option.value === current);
    select.value = keep ? current : "";
    if (current && !keep && !this.$("watchParent").checked && !this.$("watchGeneration").checked && !this.$("watchEver").checked) {
      this.$("watchParent").checked = true;
    }
  }

  saveSession() {
    try {
      const pack = (weights) => weights ? Array.from(weights, (value) => +value.toFixed(4)) : null;
      localStorage.setItem("ragdoll-panel", JSON.stringify({
        gen: this.state.gen,
        theta: pack(this.state.theta),
        bestGenTheta: pack(this.state.bestGenTheta),
        bestEverTheta: pack(this.state.bestEverTheta),
        bestEverScore: this.state.bestEverScore,
        bestEverName: this.state.bestEverName || "",
        bestEverGen: Number.isFinite(this.state.bestEverGen) ? this.state.bestEverGen : null,
        networkName: this.state.networkName || "",
        eliteScore: this.state.eliteScore,
        hist: this.state.hist.slice(-80),
        snapshots: (this.state.snapshots || []).slice(-40).map((item) => ({
          gen: item.gen, score: item.score, name: item.name || "", weights: pack(item.weights),
        })),
      }));
    } catch (error) { /* a full history can exceed the quota; the run still continues */ }
  }

  networkFromJson(data) {
    const rawArray = Array.isArray(data);
    const packet = rawArray ? { theta: data } : data;
    if (!packet || typeof packet !== "object") return null;
    const parent = ElmanNetwork.copy(packet.theta);
    if (!parent) return null;
    const revived = new EvolutionSearch(this.randomNormal);
    revived.theta = parent;
    revived.bestEverTheta = (!rawArray && ElmanNetwork.copy(packet.bestEverTheta)) || parent.slice();
    revived.bestGenTheta = (!rawArray && ElmanNetwork.copy(packet.bestGenTheta)) || parent.slice();
    revived.m = new Float32Array(parent.length);
    revived.v = new Float32Array(parent.length);
    revived.k = 0;
    if (rawArray) return revived;
    if (Number.isFinite(+packet.gen) && +packet.gen >= 0) revived.gen = Math.floor(+packet.gen);
    if (Number.isFinite(+packet.bestEverScore)) revived.bestEverScore = +packet.bestEverScore;
    if (typeof packet.bestEverName === "string") revived.bestEverName = packet.bestEverName.replace(/\.json$/i, "");
    if (Number.isFinite(+packet.bestEverGen) && +packet.bestEverGen >= 0) revived.bestEverGen = Math.floor(+packet.bestEverGen);
    if (typeof packet.name === "string") revived.networkName = packet.name.replace(/\.json$/i, "");
    if (Number.isFinite(+packet.eliteScore)) revived.eliteScore = +packet.eliteScore;
    if (Array.isArray(packet.hist)) {
      revived.hist = packet.hist.filter((item) => item && Number.isFinite(+item.mean) && Number.isFinite(+item.best));
    }
    if (Array.isArray(packet.snapshots)) {
      revived.snapshots = packet.snapshots.slice(-40).map((item) => ({
        gen: item && item.gen,
        score: item && item.score,
        name: item && typeof item.name === "string" ? item.name.replace(/\.json$/i, "") : "",
        weights: item && ElmanNetwork.copy(item.weights),
      })).filter((item) => item.weights && Number.isFinite(+item.gen) && Number.isFinite(+item.score));
    }
    return revived;
  }

  loadSession() {
    try {
      const saved = JSON.parse(localStorage.getItem("ragdoll-panel") || "null");
      if (!saved || !saved.theta) return null;
      const parent = ElmanNetwork.copy(saved.theta);
      if (!parent) return null;
      const revived = new EvolutionSearch(this.randomNormal);
      revived.theta = parent;
      revived.gen = saved.gen || 0;
      revived.bestGenTheta = ElmanNetwork.copy(saved.bestGenTheta) || parent.slice();
      revived.bestEverTheta = ElmanNetwork.copy(saved.bestEverTheta) || parent.slice();
      revived.bestEverScore = saved.bestEverScore ?? -1e9;
      revived.bestEverName = typeof saved.bestEverName === "string" ? saved.bestEverName : "";
      if (saved.bestEverGen != null && Number.isFinite(+saved.bestEverGen)) revived.bestEverGen = Math.floor(+saved.bestEverGen);
      revived.networkName = typeof saved.networkName === "string" ? saved.networkName : "";
      revived.eliteScore = saved.eliteScore ?? -1e9;
      revived.hist = saved.hist || [];
      revived.snapshots = (saved.snapshots || []).slice(-40).map((item) => ({
        gen: item.gen,
        score: item.score,
        name: item.name || "",
        weights: ElmanNetwork.copy(item.weights),
      })).filter((item) => item.weights);
      revived.m = new Float32Array(parent.length);
      revived.v = new Float32Array(parent.length);
      return revived;
    } catch (error) { return null; }
  }
}

new NeuralRagDollWorld();
