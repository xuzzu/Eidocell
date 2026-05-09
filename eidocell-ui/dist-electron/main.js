var __defProp = Object.defineProperty;
var __defNormalProp = (obj, key, value) => key in obj ? __defProp(obj, key, { enumerable: true, configurable: true, writable: true, value }) : obj[key] = value;
var __publicField = (obj, key, value) => __defNormalProp(obj, typeof key !== "symbol" ? key + "" : key, value);
import { app, BrowserWindow, ipcMain, dialog } from "electron";
import { fileURLToPath } from "node:url";
import path from "node:path";
import { spawn } from "node:child_process";
const PY_HOST = "127.0.0.1";
const PY_PORT = 8e3;
const isDev = !app.isPackaged;
class PythonManager {
  constructor() {
    __publicField(this, "process", null);
    __publicField(this, "isStopping", false);
  }
  getConfig() {
    if (isDev) {
      return {
        command: "poetry",
        args: ["run", "uvicorn", "main:app", "--host", PY_HOST, "--port", `${PY_PORT}`, "--reload"],
        cwd: path.join(process.env.APP_ROOT, "..", "eidocell-backend"),
        shell: true
      };
    } else {
      const exeName = process.platform === "win32" ? "eidocell-backend.exe" : "eidocell-backend";
      return {
        command: path.join(process.resourcesPath, "eidocell-backend", exeName),
        args: [],
        cwd: path.join(process.resourcesPath, "eidocell-backend"),
        shell: false
      };
    }
  }
  start() {
    var _a, _b;
    const { command, args, cwd, shell } = this.getConfig();
    console.log(`[PythonManager] Spawning: ${command} ${args.join(" ")} (cwd: ${cwd})`);
    this.process = spawn(command, args, { cwd, shell, stdio: "pipe" });
    (_a = this.process.stdout) == null ? void 0 : _a.on("data", (data) => {
      console.log(`[PY]: ${data.toString().trim()}`);
    });
    (_b = this.process.stderr) == null ? void 0 : _b.on("data", (data) => {
      console.error(`[PY ERR]: ${data.toString().trim()}`);
    });
    this.process.on("close", (code) => {
      console.log(`[PythonManager] Process exited with code ${code}`);
      this.process = null;
      if (!this.isStopping && !isDev && code !== 0) {
        console.log("[PythonManager] Process crashed. Restarting in 3s...");
        setTimeout(() => this.start(), 3e3);
      }
    });
  }
  stop() {
    var _a;
    this.isStopping = true;
    if ((_a = this.process) == null ? void 0 : _a.pid) {
      console.log("[PythonManager] Killing Python process tree...");
      import("./index-B5ulREBd.js").then((n) => n.i).then(({ default: treeKill }) => {
        treeKill(this.process.pid, "SIGTERM", (err) => {
          if (err) {
            console.error("[PythonManager] Failed to kill process tree:", err);
          } else {
            console.log("[PythonManager] Process tree killed.");
          }
        });
      }).catch(() => {
        var _a2;
        (_a2 = this.process) == null ? void 0 : _a2.kill("SIGTERM");
      });
      this.process = null;
    }
  }
}
const pythonManager = new PythonManager();
const __dirname$1 = path.dirname(fileURLToPath(import.meta.url));
process.env.APP_ROOT = path.join(__dirname$1, "..");
const VITE_DEV_SERVER_URL = process.env["VITE_DEV_SERVER_URL"];
const MAIN_DIST = path.join(process.env.APP_ROOT, "dist-electron");
const RENDERER_DIST = path.join(process.env.APP_ROOT, "dist");
process.env.VITE_PUBLIC = VITE_DEV_SERVER_URL ? path.join(process.env.APP_ROOT, "public") : RENDERER_DIST;
let win;
function createWindow() {
  win = new BrowserWindow({
    icon: path.join(process.env.VITE_PUBLIC, "electron-vite.svg"),
    webPreferences: {
      preload: path.join(__dirname$1, "preload.mjs")
    }
  });
  win.webContents.on("did-finish-load", () => {
    win == null ? void 0 : win.webContents.send("main-process-message", (/* @__PURE__ */ new Date()).toLocaleString());
  });
  if (VITE_DEV_SERVER_URL) {
    win.loadURL(VITE_DEV_SERVER_URL);
  } else {
    win.loadFile(path.join(RENDERER_DIST, "index.html"));
  }
}
app.on("window-all-closed", () => {
  if (process.platform !== "darwin") {
    app.quit();
    win = null;
  }
});
app.on("activate", () => {
  if (BrowserWindow.getAllWindows().length === 0) {
    createWindow();
  }
});
ipcMain.handle("select-directory", async () => {
  const result = await dialog.showOpenDialog({ properties: ["openDirectory"] });
  return result.filePaths[0] ?? null;
});
app.whenReady().then(() => {
  pythonManager.start();
  createWindow();
});
app.on("will-quit", () => {
  pythonManager.stop();
});
export {
  MAIN_DIST,
  RENDERER_DIST,
  VITE_DEV_SERVER_URL
};
