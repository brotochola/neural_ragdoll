// Sesión del navegador. Los pesos van como Float32Array. IndexedDB los clona al hacer put.
// La base se abre una vez. put tiene que correr en la misma vuelta, antes de que el loop mute los pesos.

const DB_NAME = "ragdoll";
const DB_VERSION = 1;
const SESSION_KEY = "panel";

let database = null;

export function storeReady() {
  return !!database;
}

function txDone(tx) {
  return new Promise((resolve, reject) => {
    tx.oncomplete = () => resolve();
    tx.onerror = () => reject(tx.error);
    tx.onabort = () => reject(tx.error || new Error("session store aborted"));
  });
}

export function openStore() {
  if (database) return Promise.resolve(database);
  return new Promise((resolve, reject) => {
    const request = indexedDB.open(DB_NAME, DB_VERSION);
    request.onupgradeneeded = () => {
      const db = request.result;
      if (!db.objectStoreNames.contains("session")) db.createObjectStore("session");
      if (!db.objectStoreNames.contains("snapshots")) db.createObjectStore("snapshots", { keyPath: "gen" });
    };
    request.onsuccess = () => {
      database = request.result;
      resolve(database);
    };
    request.onerror = () => reject(request.error);
  });
}

export function loadStored() {
  const tx = database.transaction(["session", "snapshots"], "readonly");
  const sessionReq = tx.objectStore("session").get(SESSION_KEY);
  const snapReq = tx.objectStore("snapshots").getAll();
  let session = null;
  let snapshots = [];
  sessionReq.onsuccess = () => { session = sessionReq.result || null; };
  snapReq.onsuccess = () => { snapshots = snapReq.result || []; };
  return txDone(tx).then(() => ({ session, snapshots }));
}

// Clona en esta llamada. El caller puede mutar los Float32Array en la línea siguiente.
export function putSession(record) {
  const tx = database.transaction("session", "readwrite");
  tx.objectStore("session").put(record, SESSION_KEY);
  return txDone(tx);
}

export function syncSnapshots(items) {
  const tx = database.transaction("snapshots", "readwrite");
  const store = tx.objectStore("snapshots");
  store.clear();
  for (let i = 0; i < items.length; i++) {
    const item = items[i];
    store.put({
      gen: item.gen,
      score: item.score,
      name: item.name || "",
      weights: item.weights,
    });
  }
  return txDone(tx);
}
