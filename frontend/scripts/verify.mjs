// Reproducible browser test: npm run verify (after setup and npx playwright install chromium).
import assert from "node:assert/strict";
import { spawn } from "node:child_process";
import { existsSync } from "node:fs";
import { mkdtemp, mkdir, writeFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import { resolve, dirname } from "node:path";
import { fileURLToPath } from "node:url";
import { chromium } from "playwright";

const root = resolve(dirname(fileURLToPath(import.meta.url)), "../..");
const runtime = await mkdtemp(resolve(tmpdir(), "crowdshield-browser-"));
const localPython = resolve(
  root,
  process.platform === "win32"
    ? ".venv/Scripts/python.exe"
    : ".venv/bin/python",
);
const python =
  process.env.CROWDSHIELD_PYTHON ||
  (existsSync(localPython) ? localPython : "python");
const server = spawn(
  python,
  [
    "-m",
    "uvicorn",
    "backend.app.main:app",
    "--host",
    "127.0.0.1",
    "--port",
    "8000",
  ],
  {
    cwd: root,
    env: { ...process.env, CROWDSHIELD_RUNTIME: runtime },
    stdio: ["ignore", "pipe", "pipe"],
  },
);
let serverLog = "";
server.stdout.on("data", (b) => (serverLog += b));
server.stderr.on("data", (b) => (serverLog += b));
let launchError;
server.on("error", (error) => (launchError = error));
const base = "http://127.0.0.1:8000";
const waitUntil = async (condition, timeout = 60000) => {
  const deadline = Date.now() + timeout;
  while (Date.now() < deadline) {
    if (await condition()) return;
    await new Promise((resolve) => setTimeout(resolve, 200));
  }
  throw new Error("Timed out waiting for condition");
};
let browser;
try {
  await waitUntil(async () => {
    if (launchError) throw launchError;
    if (server.exitCode !== null) throw new Error(serverLog);
    return fetch(`${base}/api/health`)
      .then((r) => r.ok)
      .catch(() => false);
  });
  browser = await chromium.launch({
    executablePath: process.env.CROWDSHIELD_BROWSER_PATH || undefined,
    args: process.env.CROWDSHIELD_BROWSER_PATH
      ? ["--no-sandbox", "--disable-dev-shm-usage"]
      : [],
    headless: true,
  });
  const page = await browser.newPage({
    viewport: { width: 1440, height: 1000 },
  });
  const errors = [];
  page.on("pageerror", (e) => errors.push(e.message));
  page.on("console", (msg) => {
    if (msg.type() === "error") errors.push(msg.text());
  });
  await page.goto(base);
  await page.getByText("Backend connected", { exact: true }).waitFor();
  await page
    .getByRole("heading", { name: "Your observation starts here" })
    .waitFor();
  await mkdir(resolve(root, "reports"), { recursive: true });
  await page.screenshot({
    path: resolve(root, "reports/browser-empty.png"),
    fullPage: true,
  });
  await page.getByRole("button", { name: "Load sample video" }).click();
  await page.getByText("UMN academic demonstration", { exact: true }).waitFor();
  await page.getByLabel("Reference capacity").fill("10");
  await page.getByLabel("Count forecasting").selectOption("lstm");
  await page
    .getByRole("button", { name: "Start analysis", exact: true })
    .click();
  let job;
  await waitUntil(async () => {
    const jobs = await fetch(`${base}/api/jobs`).then((r) => r.json());
    job = jobs[0];
    if (job?.status === "failed") throw new Error(job.error);
    return job?.latest?.timestamp >= 12;
  });
  const warningPopup = page.getByRole("alertdialog");
  await warningPopup.waitFor({ timeout: 60000 });
  await warningPopup.getByText("Abnormal condition warning").waitFor();
  await warningPopup
    .getByRole("button", { name: "Dismiss warning popup" })
    .click();
  await page.getByText("count lstm · experimental", { exact: true }).waitFor();
  await page.screenshot({
    path: resolve(root, "reports/browser-analysis.png"),
    fullPage: true,
  });
  await page.getByRole("button", { name: "Density map", exact: true }).click();
  await page.waitForFunction(() => {
    const img = document.querySelector(".video-stage img");
    return (
      img?.complete && img.naturalWidth > 0 && img.src.includes("heatmap=true")
    );
  });
  await page
    .getByRole("button", { name: "Stop analysis", exact: true })
    .click();
  await waitUntil(async () => {
    job = await fetch(`${base}/api/jobs/${job.id}`).then((r) => r.json());
    return ["stopped", "completed"].includes(job.status);
  });
  await page.getByRole("button", { name: /^Alerts/ }).click();
  await page
    .getByRole("button", { name: "Acknowledge", exact: true })
    .first()
    .click();
  await page
    .getByRole("button", { name: "Acknowledged", exact: true })
    .first()
    .waitFor();
  await page.getByRole("button", { name: "Evaluation", exact: true }).click();
  await page.getByText("20.69", { exact: true }).waitFor();
  await page.screenshot({
    path: resolve(root, "reports/browser-evaluation.png"),
    fullPage: true,
  });
  await page.getByRole("button", { name: "Datasets", exact: true }).click();
  await page
    .getByRole("heading", { name: "Sources and permitted uses" })
    .waitFor();
  await page.getByRole("button", { name: "Analyses", exact: true }).click();
  const exportResponse = await fetch(`${base}/api/jobs/${job.id}/export`);
  assert.equal(exportResponse.status, 200);
  assert.match(await exportResponse.text(), /forecast_count,forecast_method/);
  await page.setViewportSize({ width: 390, height: 844 });
  await page.getByRole("button", { name: "Toggle navigation" }).click();
  await page.getByRole("button", { name: "Overview", exact: true }).click();
  await page.screenshot({
    path: resolve(root, "reports/browser-mobile.png"),
    fullPage: true,
  });
  assert.equal(
    await page.evaluate(
      () => document.documentElement.scrollWidth > window.innerWidth,
    ),
    false,
    "Mobile overflow",
  );
  assert.deepEqual(errors, [], "Browser console/page errors");
  const report = {
    status: "passed",
    date_utc: new Date().toISOString(),
    browser: await browser.version(),
    viewport_desktop: "1440x1000",
    viewport_mobile: "390x844",
    console_errors: errors,
    checks: [
      "Empty state",
      "Load sample",
      "Start real inference",
      "LSTM output",
      "Abnormal-condition warning popup",
      "Heatmap",
      "Stop",
      "Acknowledge alert",
      "Evaluation",
      "Dataset registry",
      "CSV export",
      "Responsive layout",
    ],
    sample_count: job.processed_samples,
    note: "Functional verification, not crowd-safety accuracy evaluation",
  };
  await writeFile(
    resolve(root, "reports/browser_verification.json"),
    JSON.stringify(report, null, 2),
  );
  console.log(JSON.stringify(report, null, 2));
} finally {
  if (browser) await browser.close();
  server.kill("SIGTERM");
  console.log(`Test runtime retained at ${runtime}`);
}
