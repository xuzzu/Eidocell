var R = Object.defineProperty;
var _ = (t, e, o) => e in t ? R(t, e, { enumerable: !0, configurable: !0, writable: !0, value: o }) : t[e] = o;
var c = (t, e, o) => _(t, typeof e != "symbol" ? e + "" : e, o);
import { app as r, BrowserWindow as g, ipcMain as T, dialog as E } from "electron";
import { fileURLToPath as y } from "node:url";
import s from "node:path";
import { spawn as S } from "node:child_process";
const v = "127.0.0.1", O = 8e3, h = !r.isPackaged;
class j {
  constructor() {
    c(this, "process", null);
    c(this, "isStopping", !1);
  }
  getConfig() {
    if (h)
      return {
        command: "poetry",
        args: ["run", "uvicorn", "main:app", "--host", v, "--port", `${O}`, "--reload"],
        cwd: s.join(process.env.APP_ROOT, "..", "eidocell-backend"),
        shell: !0
      };
    {
      const e = process.platform === "win32" ? "eidocell-backend.exe" : "eidocell-backend";
      return {
        command: s.join(process.resourcesPath, "eidocell-backend", e),
        args: [],
        cwd: s.join(process.resourcesPath, "eidocell-backend"),
        shell: !1
      };
    }
  }
  start() {
    var p, d;
    const { command: e, args: o, cwd: l, shell: f } = this.getConfig();
    console.log(`[PythonManager] Spawning: ${e} ${o.join(" ")} (cwd: ${l})`), this.process = S(e, o, { cwd: l, shell: f, stdio: "pipe" }), (p = this.process.stdout) == null || p.on("data", (i) => {
      console.log(`[PY]: ${i.toString().trim()}`);
    }), (d = this.process.stderr) == null || d.on("data", (i) => {
      console.error(`[PY ERR]: ${i.toString().trim()}`);
    }), this.process.on("close", (i) => {
      console.log(`[PythonManager] Process exited with code ${i}`), this.process = null, !this.isStopping && !h && i !== 0 && (console.log("[PythonManager] Process crashed. Restarting in 3s..."), setTimeout(() => this.start(), 3e3));
    });
  }
  stop() {
    var e;
    this.isStopping = !0, (e = this.process) != null && e.pid && (console.log("[PythonManager] Killing Python process tree..."), import("./index-D7PzKsyU.js").then((o) => o.i).then(({ default: o }) => {
      o(this.process.pid, "SIGTERM", (l) => {
        l ? console.error("[PythonManager] Failed to kill process tree:", l) : console.log("[PythonManager] Process tree killed.");
      });
    }).catch(() => {
      var o;
      (o = this.process) == null || o.kill("SIGTERM");
    }), this.process = null);
  }
}
const P = new j(), m = s.dirname(y(import.meta.url));
process.env.APP_ROOT = s.join(m, "..");
const a = process.env.VITE_DEV_SERVER_URL, V = s.join(process.env.APP_ROOT, "dist-electron"), u = s.join(process.env.APP_ROOT, "dist");
process.env.VITE_PUBLIC = a ? s.join(process.env.APP_ROOT, "public") : u;
let n;
function w() {
  n = new g({
    icon: s.join(process.env.VITE_PUBLIC, "electron-vite.svg"),
    webPreferences: {
      preload: s.join(m, "preload.mjs")
    }
  }), n.webContents.on("did-finish-load", () => {
    n == null || n.webContents.send("main-process-message", (/* @__PURE__ */ new Date()).toLocaleString());
  }), a ? n.loadURL(a) : n.loadFile(s.join(u, "index.html"));
}
r.on("window-all-closed", () => {
  process.platform !== "darwin" && (r.quit(), n = null);
});
r.on("activate", () => {
  g.getAllWindows().length === 0 && w();
});
T.handle("select-directory", async () => (await E.showOpenDialog({ properties: ["openDirectory"] })).filePaths[0] ?? null);
r.whenReady().then(() => {
  P.start(), w();
});
r.on("will-quit", () => {
  P.stop();
});
export {
  V as MAIN_DIST,
  u as RENDERER_DIST,
  a as VITE_DEV_SERVER_URL
};
