const state = {
  schema: [],
};

const elements = {
  modelStatus: document.querySelector("#model-status"),
  setupNotice: document.querySelector("#setup-notice"),
  form: document.querySelector("#scoring-form"),
  fields: document.querySelector("#feature-fields"),
  submitButton: document.querySelector("#submit-button"),
  fillDefaultsButton: document.querySelector("#fill-defaults"),
  emptyState: document.querySelector("#empty-state"),
  result: document.querySelector("#result"),
  errorBox: document.querySelector("#error-box"),
  ring: document.querySelector("#probability-ring"),
  probability: document.querySelector("#probability-value"),
  riskBadge: document.querySelector("#risk-badge"),
  score: document.querySelector("#score-value"),
  decision: document.querySelector("#decision-value"),
  factors: document.querySelector("#factor-list"),
};

const riskLabels = {
  low: "Низкий риск",
  medium: "Средний риск",
  high: "Высокий риск",
};

const decisionLabels = {
  approve: "Можно одобрить",
  manual_review: "Ручная проверка",
  decline: "Рекомендуется отказ",
};

async function requestJson(url, options = {}) {
  const response = await fetch(url, options);
  const data = await response.json().catch(() => ({}));
  if (!response.ok) {
    throw new Error(data.detail || "Сервис временно недоступен");
  }
  return data;
}

function setModelStatus(loaded) {
  elements.modelStatus.classList.toggle("ready", loaded);
  elements.modelStatus.classList.toggle("offline", !loaded);
  elements.modelStatus.querySelector("span:last-child").textContent = loaded
    ? "Модель загружена"
    : "Модель не обучена";
}

function createField(feature) {
  const wrapper = document.createElement("div");
  wrapper.className = "field";

  const label = document.createElement("label");
  label.htmlFor = `feature-${feature.name}`;
  label.textContent = feature.label || feature.name;

  let control;
  if (feature.type === "select" && Array.isArray(feature.options)) {
    control = document.createElement("select");
    const emptyOption = document.createElement("option");
    emptyOption.value = "";
    emptyOption.textContent = "Не указано";
    control.append(emptyOption);

    feature.options.forEach((value) => {
      const option = document.createElement("option");
      option.value = value;
      option.textContent = value;
      control.append(option);
    });
  } else {
    control = document.createElement("input");
    control.type = feature.type === "number" ? "number" : "text";
    if (feature.type === "number") {
      control.step = "any";
      if (Number.isFinite(feature.min)) control.min = feature.min;
      if (Number.isFinite(feature.max)) control.max = feature.max;
    }
  }

  control.id = `feature-${feature.name}`;
  control.name = feature.name;
  control.dataset.default = feature.default ?? "";
  if (feature.required) control.required = true;

  wrapper.append(label, control);
  return wrapper;
}

function renderSchema(features) {
  elements.fields.replaceChildren();
  features.forEach((feature) => elements.fields.append(createField(feature)));
  elements.submitButton.disabled = features.length === 0;
}

function fillDefaults() {
  state.schema.forEach((feature) => {
    const control = elements.form.elements.namedItem(feature.name);
    if (control && feature.default !== null && feature.default !== undefined) {
      control.value = String(feature.default);
    }
  });
}

function readFormValues() {
  const values = {};
  state.schema.forEach((feature) => {
    const control = elements.form.elements.namedItem(feature.name);
    if (!control || control.value === "") return;
    values[feature.name] = feature.type === "number"
      ? Number(control.value)
      : control.value;
  });
  return values;
}

function renderFactors(factors) {
  elements.factors.replaceChildren();
  if (!factors.length) {
    const item = document.createElement("p");
    item.className = "factor-name";
    item.textContent = "Для этой модели детализация факторов недоступна.";
    elements.factors.append(item);
    return;
  }

  factors.forEach((factor) => {
    const item = document.createElement("div");
    item.className = "factor-item";

    const name = document.createElement("span");
    name.className = "factor-name";
    name.textContent = factor.feature;

    const direction = document.createElement("span");
    const increasesRisk = factor.direction === "increases_risk";
    direction.className = `factor-direction ${increasesRisk ? "up" : "down"}`;
    direction.textContent = increasesRisk ? "повышает риск ↑" : "снижает риск ↓";

    item.append(name, direction);
    elements.factors.append(item);
  });
}

function renderResult(data) {
  const percent = Math.round(data.default_probability * 1000) / 10;
  elements.probability.textContent = `${percent}%`;
  elements.ring.style.setProperty("--progress", `${data.default_probability * 360}deg`);
  elements.score.textContent = data.score;
  elements.riskBadge.textContent = riskLabels[data.risk_level] || data.risk_level;
  elements.riskBadge.className = `risk-badge ${data.risk_level}`;
  elements.decision.textContent = decisionLabels[data.decision] || data.decision;
  renderFactors(data.top_factors || []);

  elements.emptyState.classList.add("hidden");
  elements.errorBox.classList.add("hidden");
  elements.result.classList.remove("hidden");
}

function showError(message) {
  elements.errorBox.textContent = message;
  elements.errorBox.classList.remove("hidden");
}

async function initialize() {
  try {
    const health = await requestJson("/health");
    setModelStatus(health.model_loaded);
    if (!health.model_loaded) {
      elements.setupNotice.classList.remove("hidden");
      return;
    }

    const schema = await requestJson("/api/v1/schema");
    state.schema = schema.features || [];
    renderSchema(state.schema);
  } catch (error) {
    setModelStatus(false);
    showError(error.message);
  }
}

elements.fillDefaultsButton.addEventListener("click", fillDefaults);

elements.form.addEventListener("submit", async (event) => {
  event.preventDefault();
  elements.submitButton.disabled = true;
  elements.submitButton.textContent = "Считаем…";
  elements.errorBox.classList.add("hidden");

  try {
    const data = await requestJson("/api/v1/predict", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ features: readFormValues() }),
    });
    renderResult(data);
  } catch (error) {
    showError(error.message);
  } finally {
    elements.submitButton.disabled = false;
    elements.submitButton.textContent = "Рассчитать риск";
  }
});

initialize();
