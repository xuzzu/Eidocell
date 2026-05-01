import require$$0 from "child_process";
function _mergeNamespaces(n, m) {
  for (var i = 0; i < m.length; i++) {
    const e = m[i];
    if (typeof e !== "string" && !Array.isArray(e)) {
      for (const k in e) {
        if (k !== "default" && !(k in n)) {
          const d = Object.getOwnPropertyDescriptor(e, k);
          if (d) {
            Object.defineProperty(n, k, d.get ? d : {
              enumerable: true,
              get: () => e[k]
            });
          }
        }
      }
    }
  }
  return Object.freeze(Object.defineProperty(n, Symbol.toStringTag, { value: "Module" }));
}
function getDefaultExportFromCjs(x) {
  return x && x.__esModule && Object.prototype.hasOwnProperty.call(x, "default") ? x["default"] : x;
}
var childProcess = require$$0;
var spawn = childProcess.spawn;
var exec = childProcess.exec;
var treeKill = function(pid, signal, callback) {
  if (typeof signal === "function" && callback === void 0) {
    callback = signal;
    signal = void 0;
  }
  pid = parseInt(pid);
  if (Number.isNaN(pid)) {
    if (callback) {
      return callback(new Error("pid must be a number"));
    } else {
      throw new Error("pid must be a number");
    }
  }
  var tree = {};
  var pidsToProcess = {};
  tree[pid] = [];
  pidsToProcess[pid] = 1;
  switch (process.platform) {
    case "win32":
      exec("taskkill /pid " + pid + " /T /F", callback);
      break;
    case "darwin":
      buildProcessTree(pid, tree, pidsToProcess, function(parentPid) {
        return spawn("pgrep", ["-P", parentPid]);
      }, function() {
        killAll(tree, signal, callback);
      });
      break;
    default:
      buildProcessTree(pid, tree, pidsToProcess, function(parentPid) {
        return spawn("ps", ["-o", "pid", "--no-headers", "--ppid", parentPid]);
      }, function() {
        killAll(tree, signal, callback);
      });
      break;
  }
};
function killAll(tree, signal, callback) {
  var killed = {};
  try {
    Object.keys(tree).forEach(function(pid) {
      tree[pid].forEach(function(pidpid) {
        if (!killed[pidpid]) {
          killPid(pidpid, signal);
          killed[pidpid] = 1;
        }
      });
      if (!killed[pid]) {
        killPid(pid, signal);
        killed[pid] = 1;
      }
    });
  } catch (err) {
    if (callback) {
      return callback(err);
    } else {
      throw err;
    }
  }
  if (callback) {
    return callback();
  }
}
function killPid(pid, signal) {
  try {
    process.kill(parseInt(pid, 10), signal);
  } catch (err) {
    if (err.code !== "ESRCH") throw err;
  }
}
function buildProcessTree(parentPid, tree, pidsToProcess, spawnChildProcessesList, cb) {
  var ps = spawnChildProcessesList(parentPid);
  var allData = "";
  ps.stdout.on("data", function(data) {
    var data = data.toString("ascii");
    allData += data;
  });
  var onClose = function(code) {
    delete pidsToProcess[parentPid];
    if (code != 0) {
      if (Object.keys(pidsToProcess).length == 0) {
        cb();
      }
      return;
    }
    allData.match(/\d+/g).forEach(function(pid) {
      pid = parseInt(pid, 10);
      tree[parentPid].push(pid);
      tree[pid] = [];
      pidsToProcess[pid] = 1;
      buildProcessTree(pid, tree, pidsToProcess, spawnChildProcessesList, cb);
    });
  };
  ps.on("close", onClose);
}
const index = /* @__PURE__ */ getDefaultExportFromCjs(treeKill);
const index$1 = /* @__PURE__ */ _mergeNamespaces({
  __proto__: null,
  default: index
}, [treeKill]);
export {
  index$1 as i
};
