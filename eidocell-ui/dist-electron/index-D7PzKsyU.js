import w from "child_process";
function d(e, o) {
  for (var r = 0; r < o.length; r++) {
    const n = o[r];
    if (typeof n != "string" && !Array.isArray(n)) {
      for (const t in n)
        if (t !== "default" && !(t in e)) {
          const f = Object.getOwnPropertyDescriptor(n, t);
          f && Object.defineProperty(e, t, f.get ? f : {
            enumerable: !0,
            get: () => n[t]
          });
        }
    }
  }
  return Object.freeze(Object.defineProperty(e, Symbol.toStringTag, { value: "Module" }));
}
function p(e) {
  return e && e.__esModule && Object.prototype.hasOwnProperty.call(e, "default") ? e.default : e;
}
var h = w, c = h.spawn, O = h.exec, y = function(e, o, r) {
  if (typeof o == "function" && r === void 0 && (r = o, o = void 0), e = parseInt(e), Number.isNaN(e)) {
    if (r)
      return r(new Error("pid must be a number"));
    throw new Error("pid must be a number");
  }
  var n = {}, t = {};
  switch (n[e] = [], t[e] = 1, process.platform) {
    case "win32":
      O("taskkill /pid " + e + " /T /F", r);
      break;
    case "darwin":
      a(e, n, t, function(f) {
        return c("pgrep", ["-P", f]);
      }, function() {
        s(n, o, r);
      });
      break;
    default:
      a(e, n, t, function(f) {
        return c("ps", ["-o", "pid", "--no-headers", "--ppid", f]);
      }, function() {
        s(n, o, r);
      });
      break;
  }
};
function s(e, o, r) {
  var n = {};
  try {
    Object.keys(e).forEach(function(t) {
      e[t].forEach(function(f) {
        n[f] || (l(f, o), n[f] = 1);
      }), n[t] || (l(t, o), n[t] = 1);
    });
  } catch (t) {
    if (r)
      return r(t);
    throw t;
  }
  if (r)
    return r();
}
function l(e, o) {
  try {
    process.kill(parseInt(e, 10), o);
  } catch (r) {
    if (r.code !== "ESRCH") throw r;
  }
}
function a(e, o, r, n, t) {
  var f = n(e), i = "";
  f.stdout.on("data", function(u) {
    var u = u.toString("ascii");
    i += u;
  });
  var m = function(v) {
    if (delete r[e], v != 0) {
      Object.keys(r).length == 0 && t();
      return;
    }
    i.match(/\d+/g).forEach(function(u) {
      u = parseInt(u, 10), o[e].push(u), o[u] = [], r[u] = 1, a(u, o, r, n, t);
    });
  };
  f.on("close", m);
}
const b = /* @__PURE__ */ p(y), j = /* @__PURE__ */ d({
  __proto__: null,
  default: b
}, [y]);
export {
  j as i
};
