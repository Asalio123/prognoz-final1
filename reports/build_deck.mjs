// Сборка деки: presentation.html → presentation.pdf (playwright-core +
// системный chrome-headless-shell; браузер не скачиваем, шрифты локальные
// из reports/fonts/). Запуск: node build_deck.mjs  (или make deck)
import { chromium } from 'playwright-core';
import { existsSync, readdirSync } from 'fs';
import { join, dirname } from 'path';
import { homedir } from 'os';
import { fileURLToPath } from 'url';

const HERE = dirname(fileURLToPath(import.meta.url));
function findExecutable() {
  if (process.env.PLAYWRIGHT_CHROMIUM_EXECUTABLE_PATH && existsSync(process.env.PLAYWRIGHT_CHROMIUM_EXECUTABLE_PATH)) {
    return process.env.PLAYWRIGHT_CHROMIUM_EXECUTABLE_PATH;
  }
  if (process.env.CHROME_PATH && existsSync(process.env.CHROME_PATH)) {
    return process.env.CHROME_PATH;
  }

  // Поиск в кэше ms-playwright (macOS, Linux, Windows)
  const playwrightCacheDirs = [
    join(homedir(), 'Library/Caches/ms-playwright'),
    join(homedir(), '.cache/ms-playwright'),
    process.env.LOCALAPPDATA ? join(process.env.LOCALAPPDATA, 'ms-playwright') : null,
  ].filter(Boolean);

  for (const pDir of playwrightCacheDirs) {
    if (!existsSync(pDir)) continue;
    try {
      const entries = readdirSync(pDir);
      // Ищем headless-shell или chromium
      const shellDirs = entries.filter(d => d.startsWith('chromium_headless_shell') || d.startsWith('chromium-'));
      for (const sd of shellDirs) {
        const fullSd = join(pDir, sd);
        const subdirs = readdirSync(fullSd);
        for (const sub of subdirs) {
          const candidateFiles = [
            join(fullSd, sub, 'chrome-headless-shell'),
            join(fullSd, sub, 'chrome-headless-shell.exe'),
            join(fullSd, sub, 'chrome-mac/Chromium.app/Contents/MacOS/Chromium'),
            join(fullSd, sub, 'chrome-linux/chrome'),
            join(fullSd, sub, 'chrome-win/chrome.exe'),
            join(fullSd, 'chrome-headless-shell'),
          ];
          for (const cand of candidateFiles) {
            if (existsSync(cand)) return cand;
          }
        }
      }
    } catch {}
  }

  // Поиск системных браузеров
  const systemPaths = [
    '/Applications/Google Chrome.app/Contents/MacOS/Google Chrome',
    '/Applications/Chromium.app/Contents/MacOS/Chromium',
    '/usr/bin/google-chrome',
    '/usr/bin/google-chrome-stable',
    '/usr/bin/chromium',
    '/usr/bin/chromium-browser',
  ];
  for (const sp of systemPaths) {
    if (existsSync(sp)) return sp;
  }

  return null;
}

const exe = findExecutable();
const launchOpts = exe ? { executablePath: exe } : {};
const browser = await chromium.launch(launchOpts);
const page = await browser.newPage({ viewport: { width: 1600, height: 900 } });
await page.goto('file://' + join(HERE, 'presentation.html'), { waitUntil: 'networkidle' });
await page.evaluate(() => document.fonts.ready);
await page.waitForTimeout(600);
const light = process.argv.includes('--light');
if (light) await page.evaluate(() =>
  document.querySelectorAll('.slide').forEach(s => s.classList.add('lite')));
await page.pdf({
  path: join(HERE, light ? 'presentation_print.pdf' : 'presentation.pdf'),
  width: '1600px', height: '900px',
  printBackground: true, margin: { top: 0, bottom: 0, left: 0, right: 0 },
});
console.log(light ? 'presentation_print.pdf собран (все слайды светлые)' : 'presentation.pdf собран');
await browser.close();
