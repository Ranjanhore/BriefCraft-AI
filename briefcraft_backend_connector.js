(function BriefCraftBackendConnector() {
  if (window.__BRIEFCRAFT_BACKEND_CONNECTOR__) return;
  window.__BRIEFCRAFT_BACKEND_CONNECTOR__ = true;

  var API_BASE = String(
    window.API_BASE ||
      window.BRIEFCRAFT_API_BASE ||
      window.SC_API ||
      "https://briefcraft-ai.onrender.com"
  ).replace(/\/+$/, "");

  var BLENDER_API_BASE = String(
    window.BRIEFCRAFT_BLENDER_API_BASE ||
      window.BLENDER_API_BASE ||
      (API_BASE.indexOf("briefcraft-ai.onrender.com") !== -1 ? "https://briefcraft-ai-1.onrender.com" : API_BASE)
  ).replace(/\/+$/, "");

  var state = {
    agents: [],
    account: null,
    packages: [],
    balance: 0,
    unit: "tokens",
  };

  function token() {
    try {
      return (
        (window.bcReadToken && window.bcReadToken()) ||
        localStorage.getItem("briefcraft_access_token") ||
        localStorage.getItem("access_token") ||
        localStorage.getItem("token") ||
        localStorage.getItem("briefcraft_token") ||
        sessionStorage.getItem("briefcraft_access_token") ||
        sessionStorage.getItem("access_token") ||
        sessionStorage.getItem("token") ||
        sessionStorage.getItem("briefcraft_token") ||
        ""
      ).replace(/^Bearer\s+/i, "");
    } catch (_) {
      return "";
    }
  }

  function userId() {
    try {
      var raw = localStorage.getItem("current_user") || localStorage.getItem("user");
      if (raw) {
        var u = JSON.parse(raw);
        return u.id || u.user_id || u.email || "";
      }
    } catch (_) {}
    return "";
  }

  function headers(json) {
    var h = { Accept: "application/json" };
    if (json !== false) h["Content-Type"] = "application/json";
    var t = token();
    if (t) h.Authorization = "Bearer " + t;
    var uid = userId();
    if (uid) h["X-User-Id"] = uid;
    return h;
  }

  async function api(path, options) {
    var res = await fetch(API_BASE + path, Object.assign({ headers: headers(true) }, options || {}));
    var text = await res.text();
    var data = {};
    try {
      data = text ? JSON.parse(text) : {};
    } catch (_) {
      data = { raw: text };
    }
    if (!res.ok || data.ok === false) {
      var msg = (data.detail && (data.detail.message || data.detail)) || data.message || data.error || text || "Request failed";
      var err = new Error(typeof msg === "string" ? msg : JSON.stringify(msg));
      err.status = res.status;
      err.data = data;
      throw err;
    }
    return data;
  }

  async function blenderApi(path, options) {
    var res = await fetch(BLENDER_API_BASE + path, Object.assign({ headers: headers(true) }, options || {}));
    var text = await res.text();
    var data = {};
    try {
      data = text ? JSON.parse(text) : {};
    } catch (_) {
      data = { raw: text };
    }
    if (!res.ok || data.ok === false) {
      var msg = (data.detail && (data.detail.message || data.detail)) || data.message || data.error || text || "Request failed";
      throw new Error(typeof msg === "string" ? msg : JSON.stringify(msg));
    }
    return data;
  }

  function money(p) {
    if (p == null) return "Custom";
    try {
      return "INR " + Number(p).toLocaleString("en-IN");
    } catch (_) {
      return "INR " + p;
    }
  }

  function esc(s) {
    return String(s == null ? "" : s)
      .replace(/&/g, "&amp;")
      .replace(/</g, "&lt;")
      .replace(/>/g, "&gt;")
      .replace(/"/g, "&quot;")
      .replace(/'/g, "&#39;");
  }

  function toast(msg, type) {
    if (typeof window.showToast === "function") window.showToast(msg, type || "");
    else console.log(msg);
  }

  function currentProjectId() {
    try {
      var s = window.S || {};
      var vals = [s.currentProjectId, s.project_id, s.projectId, s.activeProjectId, s.project && (s.project.id || s.project.project_id)];
      vals.push(localStorage.getItem("briefcraft_last_project_id"), localStorage.getItem("currentProjectId"), localStorage.getItem("project_id"), localStorage.getItem("projectId"));
      for (var i = 0; i < vals.length; i++) {
        var v = String(vals[i] || "").trim();
        if (v) return v;
      }
    } catch (_) {}
    return "";
  }

  function render3DButton() {
    var existing = document.getElementById("bc3dOpenBtn");
    if (existing) return existing;
    var btn = document.createElement("button");
    btn.id = "bc3dOpenBtn";
    btn.type = "button";
    btn.className = "tb-btn tb-action outline-gold";
    btn.textContent = "3D Review";
    btn.onclick = function () {
      open3DReview();
    };
    var actions = document.querySelector(".tb-actions") || document.querySelector(".canvas-tabs") || document.body;
    actions.insertBefore(btn, actions.firstChild);
    return btn;
  }

  function inject3DTabTools() {
    render3DButton();
    document.addEventListener(
      "click",
      function (ev) {
        var t = ev.target && ev.target.closest && ev.target.closest(".ct-tab,button,a,[role='button']");
        if (!t) return;
        var txt = String(t.getAttribute("data-tab") || t.textContent || "");
        if (/3d\s*renders?|3d\s*review/i.test(txt)) setTimeout(render3DInlinePrompt, 120);
      },
      true
    );
  }

  function render3DInlinePrompt() {
    var root = document.getElementById("canvasScroll") || document.querySelector(".canvas-scroll");
    if (!root || document.getElementById("bc3dInlinePrompt")) return;
    var box = document.createElement("div");
    box.id = "bc3dInlinePrompt";
    box.style.cssText = "border:1px solid rgba(201,168,76,.35);border-radius:12px;padding:14px;margin:10px 0 14px;background:rgba(201,168,76,.08);display:flex;align-items:center;justify-content:space-between;gap:12px";
    box.innerHTML = '<div><div style="font-family:Syne,system-ui,sans-serif;font-weight:900;font-size:13px">Live 3D Review Room</div><div style="font-size:11px;color:var(--text2,#cfd5e6);margin-top:4px">Open a full-screen review room to rotate the model, inspect dimensions, compare environment and element scenes, then start Blender render.</div></div><button class="bc3d-btn primary" type="button">Open Full Screen 3D</button>';
    box.querySelector("button").onclick = open3DReview;
    root.insertBefore(box, root.firstChild);
  }

  function open3DReview(projectId) {
    ensure3DStyles();
    projectId = projectId || currentProjectId();
    if (!projectId) {
      projectId = prompt("Paste project_id for 3D review:");
      if (!projectId) return;
    }
    var old = document.getElementById("bc3dModal");
    if (old) old.remove();
    var modal = document.createElement("div");
    modal.id = "bc3dModal";
    modal.className = "bc3d-modal";
    modal.innerHTML =
      '<div class="bc3d-top"><div class="bc3d-chip">Live 3D</div><div class="bc3d-title">3D Environment + Element Review</div><button class="bc3d-btn" data-action="dims">Dimensions</button><button class="bc3d-btn" data-action="refresh">Refresh Outputs</button><button class="bc3d-btn primary" data-action="render">Start Render</button><button class="bc3d-btn" data-action="close">Close</button></div>' +
      '<div class="bc3d-body"><main class="bc3d-view"><div id="bc3dCanvas" class="bc3d-canvas"></div><div id="bc3dDims"></div><div class="bc3d-hud"><span>Drag to rotate</span><span>Wheel to zoom</span><span id="bc3dHudMode">Merged scene</span></div></main><aside class="bc3d-side"><div class="bc3d-section"><h3>Scene Mode</h3><div class="bc3d-mode"><button data-mode="environment">Environment</button><button data-mode="element">Element</button><button class="active" data-mode="merged">Merged</button></div></div><div class="bc3d-section"><h3>Element Dimensions</h3><div id="bc3dInfo"></div></div><div class="bc3d-section"><h3>Render Progress</h3><div id="bc3dStatus">Ready for review.</div><div class="bc3d-progress"><span id="bc3dProgress"></span></div></div><div class="bc3d-section"><h3>Rendered Outputs</h3><div id="bc3dGallery" class="bc3d-gallery"><div class="bc3d-empty">No render outputs loaded yet.</div></div></div></aside></div>';
    document.body.appendChild(modal);
    var viewer = create3DViewer(modal, projectId);
    modal.querySelector('[data-action="close"]').onclick = function () {
      viewer.dispose();
      modal.remove();
    };
    modal.querySelector('[data-action="dims"]').onclick = function () {
      viewer.toggleDims();
    };
    modal.querySelector('[data-action="refresh"]').onclick = function () {
      viewer.loadOutputs();
    };
    modal.querySelector('[data-action="render"]').onclick = function () {
      viewer.startRender();
    };
    modal.querySelectorAll("[data-mode]").forEach(function (b) {
      b.onclick = function () {
        modal.querySelectorAll("[data-mode]").forEach(function (x) {
          x.classList.remove("active");
        });
        b.classList.add("active");
        viewer.setMode(b.getAttribute("data-mode"));
      };
    });
  }

  function loadThree(cb) {
    if (window.THREE) return cb();
    var s = document.createElement("script");
    s.src = "https://cdnjs.cloudflare.com/ajax/libs/three.js/r128/three.min.js";
    s.onload = cb;
    s.onerror = function () {
      toast("Three.js could not load. Check internet connection.", "error");
    };
    document.head.appendChild(s);
  }

  function create3DViewer(modal, projectId) {
    var holder = modal.querySelector("#bc3dCanvas");
    var dims = modal.querySelector("#bc3dDims");
    var info = modal.querySelector("#bc3dInfo");
    var status = modal.querySelector("#bc3dStatus");
    var progress = modal.querySelector("#bc3dProgress");
    var gallery = modal.querySelector("#bc3dGallery");
    var apiState = { mode: "merged", showDims: true, renderer: null, scene: null, camera: null, group: null, raf: null, angleX: -0.4, angleY: 0.4, zoom: 76 };

    function setStatus(text, pct) {
      status.textContent = text;
      progress.style.width = Math.max(0, Math.min(100, pct || 0)) + "%";
    }

    function initThree() {
      loadThree(function () {
        var THREE = window.THREE;
        apiState.scene = new THREE.Scene();
        apiState.scene.background = new THREE.Color(0x070911);
        apiState.camera = new THREE.PerspectiveCamera(42, holder.clientWidth / Math.max(holder.clientHeight, 1), 0.1, 1000);
        apiState.camera.position.set(0, -apiState.zoom, 34);
        apiState.camera.lookAt(0, 0, 4);
        apiState.renderer = new THREE.WebGLRenderer({ antialias: true });
        apiState.renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 2));
        apiState.renderer.setSize(holder.clientWidth, holder.clientHeight);
        holder.appendChild(apiState.renderer.domElement);
        var amb = new THREE.AmbientLight(0xffffff, 0.45);
        var key = new THREE.DirectionalLight(0xf6d27a, 1.1);
        key.position.set(-12, -22, 28);
        var fill = new THREE.PointLight(0x3a6bff, 1.6, 120);
        fill.position.set(18, -18, 12);
        apiState.scene.add(amb, key, fill);
        apiState.group = new THREE.Group();
        apiState.scene.add(apiState.group);
        buildScene();
        bindControls();
        resize();
        animate();
      });
    }

    function mat(color, emissive) {
      var THREE = window.THREE;
      return new THREE.MeshStandardMaterial({ color: color, roughness: 0.42, metalness: 0.12, emissive: emissive || 0x000000, emissiveIntensity: emissive ? 0.45 : 0 });
    }

    function cube(name, size, pos, material) {
      var THREE = window.THREE;
      var mesh = new THREE.Mesh(new THREE.BoxGeometry(size[0], size[1], size[2]), material);
      mesh.name = name;
      mesh.position.set(pos[0], pos[1], pos[2]);
      apiState.group.add(mesh);
      return mesh;
    }

    function buildScene() {
      var THREE = window.THREE;
      while (apiState.group.children.length) apiState.group.remove(apiState.group.children[0]);
      var envOn = apiState.mode !== "element";
      var elementOn = apiState.mode !== "environment";
      if (envOn) {
        var grid = new THREE.GridHelper(90, 30, 0x2f3a52, 0x172033);
        grid.position.z = 0.01;
        apiState.group.add(grid);
        cube("Venue Floor 60m x 40m", [60, 40, 0.35], [0, 0, -0.18], mat(0x111827));
        cube("Registration Zone", [12, 4, 1.2], [-20, -14, 0.6], mat(0x1c2f1f));
        for (var r = 0; r < 6; r++) {
          for (var c = 0; c < 12; c++) cube("Audience Chair", [0.8, 0.8, 0.8], [(c - 5.5) * 1.7, -6 - r * 1.5, 0.4], mat(0x565a66));
        }
      }
      if (elementOn) {
        cube("Main Stage 18m x 8m x 1.2m", [18, 8, 1.2], [0, 10, 0.6], mat(0x232632));
        cube("LED Backdrop 16m x 0.35m x 6m", [16, 0.35, 6], [0, 14.2, 4.2], mat(0x123a70, 0x1a5dff));
        cube("Left Scenic Tower 2m x 1m x 7m", [2, 1, 7], [-10.5, 13.6, 3.5], mat(0x3c2a10, 0xc9a84c));
        cube("Right Scenic Tower 2m x 1m x 7m", [2, 1, 7], [10.5, 13.6, 3.5], mat(0x3c2a10, 0xc9a84c));
        cube("Hero Product Plinth 3m x 3m x 1m", [3, 3, 1], [0, 8, 1.7], mat(0x0f172a, 0xe8c97a));
      }
      renderInfo();
      renderDims();
    }

    function renderInfo() {
      var rows = [
        ["Venue", apiState.mode === "element" ? "Hidden" : "60m x 40m"],
        ["Stage", apiState.mode === "environment" ? "Hidden" : "18m x 8m x 1.2m"],
        ["LED", apiState.mode === "environment" ? "Hidden" : "16m x 6m"],
        ["Mode", apiState.mode],
        ["Project", projectId],
      ];
      info.innerHTML = rows.map(function (r) { return '<div class="bc3d-row"><b>' + esc(r[0]) + '</b><span>' + esc(r[1]) + '</span></div>'; }).join("");
      var hud = modal.querySelector("#bc3dHudMode");
      if (hud) hud.textContent = apiState.mode.charAt(0).toUpperCase() + apiState.mode.slice(1) + " scene";
    }

    function renderDims() {
      dims.innerHTML = "";
      if (!apiState.showDims) return;
      [["Stage width 18m", "44%", "40%"], ["LED height 6m", "58%", "26%"], ["Venue depth 40m", "18%", "72%"]].forEach(function (d) {
        var el = document.createElement("div");
        el.className = "bc3d-dimlabel";
        el.style.left = d[1];
        el.style.top = d[2];
        el.textContent = d[0];
        dims.appendChild(el);
      });
    }

    function bindControls() {
      var down = false, lx = 0, ly = 0;
      holder.addEventListener("pointerdown", function (e) { down = true; lx = e.clientX; ly = e.clientY; holder.setPointerCapture(e.pointerId); });
      holder.addEventListener("pointermove", function (e) { if (!down) return; apiState.angleY += (e.clientX - lx) * 0.01; apiState.angleX += (e.clientY - ly) * 0.006; lx = e.clientX; ly = e.clientY; });
      holder.addEventListener("pointerup", function () { down = false; });
      holder.addEventListener("wheel", function (e) { e.preventDefault(); apiState.zoom = Math.max(35, Math.min(130, apiState.zoom + e.deltaY * 0.04)); }, { passive: false });
      window.addEventListener("resize", resize);
    }

    function resize() {
      if (!apiState.renderer) return;
      apiState.camera.aspect = holder.clientWidth / Math.max(holder.clientHeight, 1);
      apiState.camera.updateProjectionMatrix();
      apiState.renderer.setSize(holder.clientWidth, holder.clientHeight);
    }

    function animate() {
      if (!apiState.renderer) return;
      apiState.raf = requestAnimationFrame(animate);
      apiState.group.rotation.x = apiState.angleX;
      apiState.group.rotation.z = apiState.angleY;
      apiState.camera.position.set(0, -apiState.zoom, 34);
      apiState.camera.lookAt(0, 0, 4);
      apiState.renderer.render(apiState.scene, apiState.camera);
    }

    async function startRender() {
      setStatus("Starting Blender render. This can take a minute.", 18);
      modal.querySelector('[data-action="render"]').disabled = true;
      try {
        var data = await blenderApi("/blender/render", { method: "POST", body: JSON.stringify({ project_id: projectId, concept_index: 0, width: 1280, height: 720, run_now: true }) });
        setStatus("Render completed. Loading outputs.", 100);
        renderGallery((data.render_result && data.render_result.public_outputs) || {});
      } catch (e) {
        setStatus("Render failed: " + e.message, 0);
      } finally {
        modal.querySelector('[data-action="render"]').disabled = false;
      }
    }

    async function loadOutputs() {
      setStatus("Loading stored 3D outputs.", 20);
      try {
        var data = await api("/projects/" + encodeURIComponent(projectId) + "/assets?section=renders", { method: "GET", headers: headers(false) });
        var renders = {};
        (data.assets || []).forEach(function (a, i) {
          if (a.preview_url || a.image_url) renders[a.title || "render_" + i] = { url: a.preview_url || a.image_url };
        });
        renderGallery({ renders: renders });
        setStatus("Outputs loaded.", 100);
      } catch (e) {
        setStatus("No stored outputs yet. Start render to create them.", 0);
      }
    }

    function renderGallery(outputs) {
      var renders = (outputs && outputs.renders) || {};
      var keys = Object.keys(renders);
      if (!keys.length) {
        gallery.innerHTML = '<div class="bc3d-empty">No public render URLs yet.</div>';
        return;
      }
      gallery.innerHTML = keys.map(function (k) {
        var url = renders[k].url || renders[k];
        return '<button class="bc3d-thumb" type="button" onclick="window.open(\'' + esc(url) + '\',\'_blank\',\'noopener\')"><img src="' + esc(url) + '" alt="' + esc(k) + '"><div>' + esc(k) + '</div></button>';
      }).join("");
    }

    initThree();
    setTimeout(loadOutputs, 500);
    return {
      setMode: function (mode) { apiState.mode = mode || "merged"; buildScene(); },
      toggleDims: function () { apiState.showDims = !apiState.showDims; renderDims(); },
      startRender: startRender,
      loadOutputs: loadOutputs,
      dispose: function () { if (apiState.raf) cancelAnimationFrame(apiState.raf); if (apiState.renderer) apiState.renderer.dispose(); },
    };
  }

  function ensureStyles() {
    if (document.getElementById("bc-account-agent-style")) return;
    var style = document.createElement("style");
    style.id = "bc-account-agent-style";
    style.textContent =
      ".bc-credit-chip{display:inline-flex;align-items:center;gap:8px;height:34px;padding:0 12px;border-radius:999px;border:1px solid var(--gold-mid,#c9a84c55);background:var(--gold-dim,#c9a84c1a);color:var(--gold2,#e8c97a);font-size:11px;font-weight:800;letter-spacing:.04em;white-space:nowrap;cursor:pointer}" +
      ".bc-credit-chip b{color:var(--text,#fff);font-variant-numeric:tabular-nums}" +
      ".bc-agent-panel{background:var(--surface,#0d0f18);border:1px solid var(--border2,#262840);border-radius:10px;padding:14px;margin-bottom:14px}" +
      ".bc-agent-grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(230px,1fr));gap:12px}" +
      ".bc-agent-card{background:var(--surface2,#12141f);border:1px solid var(--border2,#262840);border-radius:10px;padding:12px;display:flex;flex-direction:column;gap:8px}" +
      ".bc-agent-card h4{margin:0;font-size:12px;color:var(--text,#fff);font-family:Syne,system-ui,sans-serif}" +
      ".bc-agent-card p{margin:0;color:var(--text2,#c9c6d2);font-size:10.5px;line-height:1.5}" +
      ".bc-agent-meta{display:flex;align-items:center;justify-content:space-between;gap:8px;color:var(--text3,#8e8ba0);font-size:10px}" +
      ".bc-agent-btn,.bc-plan-btn{border:1px solid var(--gold-mid,#c9a84c55);background:var(--gold-dim,#c9a84c1a);color:var(--gold2,#e8c97a);border-radius:8px;padding:8px 10px;font-size:11px;font-weight:800;cursor:pointer}" +
      ".bc-agent-btn:hover,.bc-plan-btn:hover{border-color:var(--gold,#c9a84c)}" +
      ".bc-plan-modal{position:fixed;inset:0;background:rgba(0,0,0,.74);z-index:99999;display:flex;align-items:center;justify-content:center;padding:20px;backdrop-filter:blur(8px)}" +
      ".bc-plan-box{width:min(1120px,100%);max-height:92vh;overflow:auto;background:var(--surface,#0d0f18);border:1px solid var(--border2,#262840);border-radius:16px;padding:20px;color:var(--text,#fff);box-shadow:0 30px 90px rgba(0,0,0,.55)}" +
      ".bc-plan-head{display:flex;justify-content:space-between;align-items:flex-start;gap:16px;margin-bottom:16px}.bc-plan-head h2{margin:0;font-family:Syne,system-ui,sans-serif;font-size:24px}.bc-plan-close{width:34px;height:34px;border-radius:8px;border:1px solid var(--border2,#262840);background:transparent;color:var(--text,#fff);cursor:pointer}" +
      ".bc-plan-tabs{display:flex;gap:8px;flex-wrap:wrap;margin-bottom:14px}.bc-plan-tabs button{border:1px solid var(--border2,#262840);background:var(--surface2,#12141f);color:var(--text2,#c9c6d2);border-radius:999px;padding:8px 12px;cursor:pointer}.bc-plan-tabs button.active{border-color:var(--gold,#c9a84c);color:var(--gold2,#e8c97a);background:var(--gold-dim,#c9a84c1a)}" +
      ".bc-plan-grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(250px,1fr));gap:12px}.bc-plan-card{position:relative;border:1px solid var(--border2,#262840);background:var(--surface2,#12141f);border-radius:12px;padding:16px;display:flex;flex-direction:column;gap:10px}.bc-plan-card.recommended{border-color:var(--gold,#c9a84c);box-shadow:0 0 0 2px var(--gold-dim,#c9a84c1a)}.bc-plan-card h3{margin:0;font-family:Syne,system-ui,sans-serif;font-size:15px}.bc-plan-price{font-size:24px;font-weight:900;color:var(--gold2,#e8c97a)}.bc-plan-card ul{padding-left:18px;margin:0;color:var(--text2,#c9c6d2);font-size:11px;line-height:1.6}.bc-plan-badge{position:absolute;right:12px;top:12px;border-radius:999px;background:var(--gold,#c9a84c);color:#07080d;font-size:9px;font-weight:900;padding:4px 8px;text-transform:uppercase}";
    document.head.appendChild(style);
  }

  function ensure3DStyles() {
    if (document.getElementById("bc-3d-review-style")) return;
    var style = document.createElement("style");
    style.id = "bc-3d-review-style";
    style.textContent =
      ".bc3d-modal{position:fixed;inset:0;z-index:2147483000;background:#05070d;color:#f7f4ec;display:flex;flex-direction:column;font-family:DM Sans,system-ui,sans-serif}" +
      ".bc3d-top{height:58px;display:flex;align-items:center;gap:12px;padding:0 18px;border-bottom:1px solid rgba(255,255,255,.10);background:linear-gradient(180deg,#0c0f18,#070912)}" +
      ".bc3d-title{font-family:Syne,system-ui,sans-serif;font-size:15px;font-weight:900;flex:1}.bc3d-chip{border:1px solid rgba(201,168,76,.35);background:rgba(201,168,76,.10);color:#e8c97a;border-radius:999px;padding:6px 10px;font-size:10px;font-weight:900;text-transform:uppercase;letter-spacing:.08em}" +
      ".bc3d-btn{border:1px solid rgba(201,168,76,.38);background:rgba(201,168,76,.10);color:#f6d27a;border-radius:9px;padding:9px 12px;font-size:11px;font-weight:900;cursor:pointer}.bc3d-btn.primary{background:linear-gradient(135deg,#f6d27a,#a87518);color:#090b10;border:none}.bc3d-btn:disabled{opacity:.48;cursor:wait}" +
      ".bc3d-body{flex:1;display:grid;grid-template-columns:minmax(0,1fr) 360px;min-height:0}.bc3d-view{position:relative;overflow:hidden;background:radial-gradient(circle at 50% 20%,rgba(58,106,255,.12),transparent 35%),#070911}.bc3d-canvas{position:absolute;inset:0}.bc3d-hud{position:absolute;left:18px;bottom:18px;display:flex;gap:8px;flex-wrap:wrap}.bc3d-hud span{background:rgba(0,0,0,.55);border:1px solid rgba(255,255,255,.12);border-radius:999px;padding:7px 10px;font-size:10px;color:#dbeafe}" +
      ".bc3d-side{border-left:1px solid rgba(255,255,255,.10);background:#0b0d14;overflow:auto;padding:16px}.bc3d-section{border:1px solid rgba(255,255,255,.10);background:rgba(255,255,255,.04);border-radius:12px;padding:12px;margin-bottom:12px}.bc3d-section h3{margin:0 0 8px;font-family:Syne,system-ui,sans-serif;font-size:13px}.bc3d-row{display:grid;grid-template-columns:110px 1fr;gap:8px;font-size:11px;color:#cfd5e6;margin:6px 0}.bc3d-row b{color:#f6d27a}.bc3d-mode{display:grid;grid-template-columns:1fr 1fr 1fr;gap:6px}.bc3d-mode button{border:1px solid rgba(255,255,255,.12);background:#111827;color:#cfd5e6;border-radius:8px;padding:8px 6px;font-size:10px;font-weight:800;cursor:pointer}.bc3d-mode button.active{border-color:#f6d27a;color:#f6d27a;background:rgba(201,168,76,.12)}" +
      ".bc3d-gallery{display:grid;grid-template-columns:1fr 1fr;gap:8px}.bc3d-thumb{border:1px solid rgba(255,255,255,.12);background:#05070d;border-radius:9px;overflow:hidden;cursor:pointer}.bc3d-thumb img{width:100%;aspect-ratio:16/9;object-fit:cover;display:block}.bc3d-thumb div{padding:6px;font-size:10px;color:#cfd5e6}.bc3d-progress{height:8px;border-radius:999px;background:rgba(255,255,255,.08);overflow:hidden;margin-top:9px}.bc3d-progress span{display:block;height:100%;width:0;background:linear-gradient(90deg,#a87518,#f6d27a);transition:width .3s}" +
      ".bc3d-dimlabel{position:absolute;pointer-events:none;background:rgba(0,0,0,.68);border:1px solid rgba(246,210,122,.45);color:#f6d27a;border-radius:999px;padding:5px 8px;font-size:10px;font-weight:900}.bc3d-empty{height:160px;display:grid;place-items:center;text-align:center;color:#8790a4;font-size:12px;border:1px dashed rgba(255,255,255,.14);border-radius:10px}.bc3d-toast{position:absolute;right:18px;bottom:18px;background:#10131c;border:1px solid rgba(201,168,76,.35);border-radius:10px;color:#f6d27a;padding:10px 12px;font-size:11px;box-shadow:0 16px 50px rgba(0,0,0,.35)}" +
      "@media(max-width:900px){.bc3d-body{grid-template-columns:1fr}.bc3d-side{border-left:none;border-top:1px solid rgba(255,255,255,.10);max-height:42vh}.bc3d-top{flex-wrap:wrap;height:auto;padding:10px}}";
    document.head.appendChild(style);
  }

  function ensureCreditChip() {
    ensureStyles();
    var chip = document.getElementById("bcCreditChip");
    if (!chip) {
      chip = document.createElement("button");
      chip.id = "bcCreditChip";
      chip.className = "bc-credit-chip";
      chip.type = "button";
      chip.onclick = openPlans;
      var actions = document.querySelector(".tb-actions") || document.querySelector(".topbar") || document.body;
      actions.insertBefore(chip, actions.firstChild);
    }
    chip.innerHTML = 'Credits <b>' + esc(state.balance || 0) + '</b> ' + esc(state.unit || "tokens");
  }

  function renderAgentsPanel() {
    ensureStyles();
    var grid = document.getElementById("connGrid");
    if (!grid) return;
    var panel = document.getElementById("bcAgentsPanel");
    if (!panel) {
      panel = document.createElement("div");
      panel.id = "bcAgentsPanel";
      panel.className = "bc-agent-panel";
      grid.parentNode.insertBefore(panel, grid);
    }
    panel.innerHTML =
      '<div style="display:flex;align-items:center;justify-content:space-between;gap:12px;margin-bottom:10px">' +
      '<div><div style="font-family:Syne,system-ui,sans-serif;font-size:14px;font-weight:800">Backend Agents</div>' +
      '<div style="font-size:11px;color:var(--text2,#c9c6d2);margin-top:3px">Auto-discovered from /agents. Add a backend agent and it appears here automatically.</div></div>' +
      '<button class="bc-agent-btn" onclick="BriefCraftAccount.refresh()">Refresh</button></div>' +
      '<div class="bc-agent-grid">' +
      state.agents
        .map(function (a) {
          return (
            '<div class="bc-agent-card"><h4>' +
            esc(a.name || a.id) +
            "</h4><p>" +
            esc(a.description || "") +
            '</p><div class="bc-agent-meta"><span>' +
            esc(a.category || "agent") +
            "</span><span>" +
            esc(a.credit_cost || 0) +
            " tokens/run</span></div>" +
            '<button class="bc-agent-btn" onclick="BriefCraftAccount.runAgent(\'' +
            esc(a.id) +
            "')\">Connect / Test</button></div>"
          );
        })
        .join("") +
      "</div>";
  }

  function openPlans(audience) {
    ensureStyles();
    var old = document.getElementById("bcPlanModal");
    if (old) old.remove();
    var modal = document.createElement("div");
    modal.id = "bcPlanModal";
    modal.className = "bc-plan-modal";
    modal.innerHTML =
      '<div class="bc-plan-box"><div class="bc-plan-head"><div><h2>Credits and Packages</h2>' +
      '<div style="font-size:12px;color:var(--text2,#c9c6d2);margin-top:6px">Balance: <b style="color:var(--gold2,#e8c97a)">' +
      esc(state.balance || 0) +
      " " +
      esc(state.unit || "tokens") +
      "</b></div></div>" +
      '<button class="bc-plan-close" onclick="document.getElementById(\'bcPlanModal\').remove()">x</button></div>' +
      '<div class="bc-plan-tabs"><button data-audience="all">All</button><button data-audience="individual">Individuals</button><button data-audience="institution">Institutions</button></div>' +
      '<div class="bc-plan-grid" id="bcPlanGrid"></div></div>';
    document.body.appendChild(modal);
    modal.addEventListener("click", function (e) {
      if (e.target === modal) modal.remove();
    });
    modal.querySelectorAll("[data-audience]").forEach(function (btn) {
      btn.onclick = function () {
        renderPlans(btn.getAttribute("data-audience"));
      };
    });
    renderPlans(audience || "all");
  }

  function renderPlans(audience) {
    audience = audience || "all";
    document.querySelectorAll("#bcPlanModal [data-audience]").forEach(function (b) {
      b.classList.toggle("active", b.getAttribute("data-audience") === audience);
    });
    var grid = document.getElementById("bcPlanGrid");
    if (!grid) return;
    var plans = state.packages.filter(function (p) {
      return audience === "all" || p.audience === audience;
    });
    grid.innerHTML = plans
      .map(function (p) {
        return (
          '<div class="bc-plan-card ' +
          (p.recommended ? "recommended" : "") +
          '">' +
          (p.recommended ? '<span class="bc-plan-badge">Recommended</span>' : "") +
          "<h3>" +
          esc(p.name) +
          '</h3><div class="bc-plan-price">' +
          esc(money(p.price_inr)) +
          '</div><div style="font-size:12px;color:var(--text2,#c9c6d2)">' +
          esc((p.credits || 0).toLocaleString ? p.credits.toLocaleString("en-IN") : p.credits) +
          " tokens / " +
          esc(p.billing) +
          "</div><ul>" +
          (p.features || [])
            .map(function (f) {
              return "<li>" + esc(f) + "</li>";
            })
            .join("") +
          '</ul><button class="bc-plan-btn" onclick="BriefCraftAccount.checkout(\'' +
          esc(p.id) +
          "')\">" +
          (p.price_inr == null ? "Contact / Request" : "Upgrade and Pay") +
          "</button></div>"
        );
      })
      .join("");
  }

  async function refreshAccount() {
    try {
      var data = await api("/account/balance", { method: "GET", headers: headers(false) });
      state.account = data.account || null;
      state.balance = data.balance || (data.account && data.account.credit_balance) || 0;
      state.unit = data.unit || "tokens";
    } catch (e) {
      console.warn("[BriefCraftAccount] balance failed", e);
    }
    ensureCreditChip();
  }

  async function refreshPackages() {
    try {
      var data = await api("/account/packages", { method: "GET", headers: headers(false) });
      state.packages = data.packages || [];
    } catch (e) {
      console.warn("[BriefCraftAccount] packages failed", e);
    }
  }

  async function refreshAgents() {
    try {
      var data = await api("/agents", { method: "GET", headers: headers(false) });
      state.agents = data.agents || [];
    } catch (e) {
      console.warn("[BriefCraftAccount] agents failed", e);
    }
    renderAgentsPanel();
  }

  async function checkout(packageId) {
    try {
      var data = await api("/account/checkout", {
        method: "POST",
        body: JSON.stringify({
          package_id: packageId,
          success_url: location.href,
          cancel_url: location.href,
        }),
      });
      toast(data.message || "Checkout ready", "success");
      if (data.checkout_url && /^https?:\/\//i.test(data.checkout_url)) window.open(data.checkout_url, "_blank", "noopener");
      await refreshAccount();
      ensureCreditChip();
      openPlans();
    } catch (e) {
      toast(e.message || "Checkout failed", "error");
    }
  }

  async function runAgent(agentId, message, context) {
    try {
      var data = await api("/agents/run", {
        method: "POST",
        body: JSON.stringify({
          agent_id: agentId,
          message: message || "Connect and report status",
          project_id: (window.S && window.S.currentProjectId) || localStorage.getItem("currentProjectId") || null,
          session_id: (window.S && window.S.session_id) || null,
          context: context || {},
        }),
      });
      state.account = data.account || state.account;
      state.balance = (data.account && data.account.credit_balance) || state.balance;
      ensureCreditChip();
      var reply =
        (data.result && (data.result.message || data.result.detail)) ||
        (data.agent && data.agent.name ? data.agent.name + " connected." : "Agent connected.");
      if (typeof window.addChat === "function") window.addChat("ai", reply);
      else toast(reply, "success");
      return data;
    } catch (e) {
      if (e.status === 402) openPlans();
      toast(e.message || "Agent failed", "error");
      throw e;
    }
  }

  async function refreshAll() {
    await Promise.all([refreshAccount(), refreshPackages(), refreshAgents()]);
    ensureCreditChip();
    renderAgentsPanel();
  }

  window.BriefCraftAccount = {
    state: state,
    refresh: refreshAll,
    refreshAccount: refreshAccount,
    refreshAgents: refreshAgents,
    openPlans: openPlans,
    checkout: checkout,
    runAgent: runAgent,
    api: api,
  };

  window.BriefCraft3DReview = {
    open: open3DReview,
    renderButton: render3DButton,
    apiBase: BLENDER_API_BASE,
  };

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", function () {
      refreshAll();
      inject3DTabTools();
      render3DInlinePrompt();
    });
  } else {
    setTimeout(refreshAll, 60);
    setTimeout(function () {
      inject3DTabTools();
      render3DInlinePrompt();
    }, 120);
  }
})();
