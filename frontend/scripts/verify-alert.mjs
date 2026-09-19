import assert from "node:assert/strict";
import { existsSync } from "node:fs";
import { resolve, dirname } from "node:path";
import { fileURLToPath } from "node:url";
import { chromium } from "playwright";

const root = resolve(dirname(fileURLToPath(import.meta.url)), "../..");
const base = process.env.CROWDSHIELD_URL || "http://127.0.0.1:8000";
const alert = {
  id: "browser-verification-alert",
  job_id: "browser-verification-job",
  zone: "Main concourse",
  created_at: new Date().toISOString(),
  video_timestamp: 18.5,
  score: 86,
  tier: "Critical",
  alert_type: "Crowding / capacity",
  verification: {
    status: "confirmed",
    confidence: 0.93,
  },
  reasons: ["Observed count exceeds the configured reference capacity"],
  action: "Ask the responsible operator to review this zone.",
  acknowledged: false,
};
const secondAlert = {
  ...alert,
  id: "browser-verification-alert-2",
  zone: "East gate",
  video_timestamp: 46,
};

let browser;
try {
  const browserCandidates = [
    process.env.CROWDSHIELD_BROWSER_PATH,
    "C:/Program Files (x86)/Microsoft/Edge/Application/msedge.exe",
    "C:/Program Files/Microsoft/Edge/Application/msedge.exe",
    "C:/Program Files/Google/Chrome/Application/chrome.exe",
    "C:/Program Files (x86)/Google/Chrome/Application/chrome.exe",
  ].filter(Boolean);
  const executablePath = browserCandidates.find((path) => existsSync(path));
  browser = await chromium.launch({ executablePath, headless: true });
  const page = await browser.newPage({ viewport: { width: 1280, height: 900 } });
  const errors = [];
  let revealAlert = false;
  let revealSecondAlert = false;
  let acknowledged = false;
  page.on("pageerror", (error) => errors.push(error.message));
  page.on("console", (message) => {
    if (message.type() === "error") errors.push(message.text());
  });
  await page.route("**/api/alerts**", async (route) => {
    if (route.request().method() === "POST") {
      acknowledged = true;
      await route.fulfill({ status: 200, contentType: "application/json", body: '{"acknowledged":true}' });
      return;
    }
    const payload = revealAlert
      ? [
          ...(revealSecondAlert ? [{ ...secondAlert, acknowledged }] : []),
          { ...alert, acknowledged },
        ]
      : [];
    await route.fulfill({
      status: 200,
      contentType: "application/json",
      body: JSON.stringify(payload),
    });
  });

  await page.goto(base);
  await page.waitForFunction(
    () => document.body.innerText.includes("Backend connected"),
    null,
    { timeout: 30000 },
  );
  revealAlert = true;

  const popup = page.getByRole("alertdialog");
  await popup.waitFor({ timeout: 10000 });
  await popup.getByText("Abnormal condition warning", { exact: true }).waitFor();
  assert.match(await popup.innerText(), /Main concourse/);
  assert.match(await popup.innerText(), /Critical/i);
  assert.match(await popup.innerText(), /Crowding \/ capacity/);
  assert.match(await popup.innerText(), /OpenClaw verified/);
  assert.match(
    await popup.innerText(),
    /Observed count exceeds the configured reference capacity/,
  );

  // A dismissed but unacknowledged warning must return after a reload. This
  // covers operators opening or refreshing the dashboard after an alert was
  // already persisted by the backend.
  await popup
    .getByRole("button", { name: "Dismiss warning popup" })
    .click();
  await popup.waitFor({ state: "hidden" });
  revealSecondAlert = true;
  await page.reload();
  const restoredPopup = page.getByRole("alertdialog");
  await restoredPopup.waitFor({ timeout: 10000 });
  assert.match(await restoredPopup.innerText(), /Main concourse/);
  assert.match(await restoredPopup.innerText(), /2 warnings pending/);
  await page.screenshot({
    path: resolve(root, "reports/browser-alert-popup.png"),
    fullPage: true,
  });

  await restoredPopup
    .getByRole("button", { name: "Dismiss warning popup" })
    .click();
  await restoredPopup.getByText("East gate", { exact: true }).waitFor();
  await restoredPopup.getByRole("button", { name: "View alert history" }).click();
  await page.getByRole("heading", { name: "Alert center" }).waitFor();
  const historyRow = page.locator(".alert-row").filter({ hasText: "Main concourse" });
  await historyRow.waitFor();
  await historyRow.getByRole("button", { name: "Acknowledge", exact: true }).click();
  await historyRow.getByRole("button", { name: "Acknowledged", exact: true }).waitFor();
  assert.equal(acknowledged, true);
  assert.deepEqual(errors, []);
  console.log("Alert popup, history, and acknowledgment verification passed");
} finally {
  if (browser) await browser.close();
}
