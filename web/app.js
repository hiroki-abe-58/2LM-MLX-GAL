/* チャットUIのふるまい.
   サーバとのやりとりは /api/chat の Server-Sent Events を読むだけ。 */

const PARAM_KEYS = ["temperature", "top_k", "repetition_penalty", "max_new_tokens", "history_turns"];
const DEFAULT_PARAMS = {
  temperature: 0.8,
  top_k: 40,
  repetition_penalty: 1.15,
  max_new_tokens: 200,
  history_turns: 2,
};

const $ = (id) => document.getElementById(id);
const el = {
  thread: $("thread"),
  welcome: $("welcome"),
  form: $("composer"),
  input: $("input"),
  send: $("btn-send"),
  clear: $("btn-clear"),
  settings: $("btn-settings"),
  meta: $("model-meta"),
  backdrop: $("modal-backdrop"),
  modalClose: $("modal-close"),
  modalApply: $("modal-apply"),
  modalReset: $("modal-reset"),
  spec: $("spec"),
  toast: $("toast"),
};

const state = {
  params: { ...DEFAULT_PARAMS },
  history: [],
  busy: false,
  info: null,
};

// リセット時に出し直せるよう、初期表示のHTMLを取っておく
const welcomeHTML = el.welcome.outerHTML;

const icons = () => window.lucide?.createIcons();

function toast(message, ms = 3200) {
  el.toast.textContent = message;
  el.toast.hidden = false;
  clearTimeout(toast._t);
  toast._t = setTimeout(() => { el.toast.hidden = true; }, ms);
}

/* ---------- メッセージ描画 ---------- */

function addMessage(role, text) {
  el.thread.querySelector(".welcome")?.remove();
  const wrap = document.createElement("div");
  wrap.className = `msg ${role}`;
  wrap.innerHTML = `
    <div class="avatar"><i data-lucide="${role === "user" ? "user-round" : "sparkles"}"></i></div>
    <div class="bubble"></div>`;
  const bubble = wrap.querySelector(".bubble");
  if (text) bubble.textContent = text;
  el.thread.appendChild(wrap);
  icons();
  scrollToEnd();
  return bubble;
}

function scrollToEnd() {
  el.thread.scrollTo({ top: el.thread.scrollHeight, behavior: "smooth" });
}

function setBusy(busy) {
  state.busy = busy;
  el.send.disabled = busy;
}

/* ---------- 送信 ---------- */

async function send(message) {
  if (state.busy || !message.trim()) return;
  const text = message.trim();
  addMessage("user", text);
  el.input.value = "";
  autoGrow();
  setBusy(true);

  const bubble = addMessage("bot", "");
  bubble.innerHTML = '<span class="thinking"><i></i><i></i><i></i></span>';
  let reply = "";
  let first = true;

  try {
    const res = await fetch("/api/chat", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ message: text, history: state.history, ...state.params }),
    });
    if (!res.ok) throw new Error(`HTTP ${res.status}`);

    const reader = res.body.getReader();
    const decoder = new TextDecoder();
    let buffer = "";

    while (true) {
      const { value, done } = await reader.read();
      if (done) break;
      buffer += decoder.decode(value, { stream: true });
      // SSE は空行でイベントが区切られる。途中で切れた分は buffer に残す。
      const events = buffer.split("\n\n");
      buffer = events.pop();
      for (const event of events) {
        const line = event.split("\n").find((l) => l.startsWith("data:"));
        if (!line) continue;
        const payload = JSON.parse(line.slice(5));
        if (payload.t !== undefined) {
          if (first) { bubble.textContent = ""; first = false; }
          reply += payload.t;
          bubble.textContent = reply;
          bubble.insertAdjacentHTML("beforeend", '<span class="caret"></span>');
          scrollToEnd();
        } else if (payload.done) {
          bubble.textContent = reply || "(何も出力されませんでした)";
          const stats = document.createElement("div");
          stats.className = "stats";
          stats.textContent =
            `${payload.stats.chars} 文字 / ${payload.stats.seconds} 秒 / ${payload.stats.cps} 文字毎秒`;
          bubble.appendChild(stats);
        }
      }
    }
    state.history.push([text, reply]);
  } catch (err) {
    bubble.textContent = "サーバに接続できませんでした。";
    toast(`エラー: ${err.message}. server.py が動いているか確認してください。`);
  } finally {
    setBusy(false);
    el.input.focus();
  }
}

/* ---------- 入力欄 ---------- */

function autoGrow() {
  el.input.style.height = "auto";
  el.input.style.height = `${Math.min(el.input.scrollHeight, 148)}px`;
}

el.input.addEventListener("input", autoGrow);
el.input.addEventListener("keydown", (e) => {
  if (e.key === "Enter" && !e.shiftKey && !e.isComposing) {
    e.preventDefault();
    send(el.input.value);
  }
});
el.form.addEventListener("submit", (e) => {
  e.preventDefault();
  send(el.input.value);
});
document.addEventListener("click", (e) => {
  const chip = e.target.closest(".chip");
  if (chip) send(chip.textContent);
});

el.clear.addEventListener("click", () => {
  state.history = [];
  el.thread.innerHTML = welcomeHTML;
  icons();
  toast("会話履歴をリセットしました");
});

/* ---------- 設定モーダル ---------- */

function syncSliders() {
  for (const key of PARAM_KEYS) {
    const input = $(key);
    input.value = state.params[key];
    updateOutput(key);
  }
}

function updateOutput(key) {
  const value = Number($(key).value);
  $(`out-${key}`).textContent = Number.isInteger(DEFAULT_PARAMS[key])
    ? value
    : value.toFixed(2);
}

for (const key of PARAM_KEYS) {
  $(key).addEventListener("input", () => updateOutput(key));
}

function openModal() {
  syncSliders();
  el.backdrop.hidden = false;
  el.backdrop.classList.remove("closing");
}

function closeModal() {
  el.backdrop.classList.add("closing");
  setTimeout(() => {
    el.backdrop.hidden = true;
    el.backdrop.classList.remove("closing");
  }, 220);
}

el.settings.addEventListener("click", openModal);
el.modalClose.addEventListener("click", closeModal);
el.backdrop.addEventListener("click", (e) => {
  if (e.target === el.backdrop) closeModal();
});
document.addEventListener("keydown", (e) => {
  if (e.key === "Escape" && !el.backdrop.hidden) closeModal();
});
el.modalReset.addEventListener("click", () => {
  state.params = { ...DEFAULT_PARAMS };
  syncSliders();
});
el.modalApply.addEventListener("click", () => {
  for (const key of PARAM_KEYS) {
    const value = Number($(key).value);
    state.params[key] = Number.isInteger(DEFAULT_PARAMS[key]) ? Math.round(value) : value;
  }
  closeModal();
  toast("生成設定を適用しました");
});

/* ---------- モデル情報 ---------- */

// --port を変えても案内が食い違わないように、実際に開いている URL を出す
document.querySelectorAll(".origin").forEach((node) => {
  node.textContent = location.origin;
});

async function loadInfo() {
  try {
    const info = await (await fetch("/api/info")).json();
    state.info = info;
    el.meta.textContent =
      `${info.params_m}M params / 語彙 ${info.vocab_size} / 文脈 ${info.block_size}文字`;
    el.spec.textContent = [
      `checkpoint : ${info.checkpoint}`,
      `layers     : ${info.n_layer}`,
      `heads      : ${info.n_head}`,
      `d_model    : ${info.n_embd}`,
      `block_size : ${info.block_size}`,
      `vocab_size : ${info.vocab_size}`,
    ].join("\n");
  } catch (err) {
    el.meta.textContent = "サーバに接続できません";
    toast("server.py が起動していないようです。");
  }
}

icons();
loadInfo();
el.input.focus();
