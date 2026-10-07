// Сборка деки: presentation.html → presentation.pdf (playwright-core +
// системный chrome-headless-shell; браузер не скачиваем, шрифты локальные
// из reports/fonts/). Запуск: node build_deck.mjs  (или make deck)
import { chromium } from 'playwright-core';
import { existsSync, readdirSync } from 'fs';
import { join, dirname } from 'path';
import { homedir } from 'os';
import { fileURLToPath } from 'url';

const HERE = dirname(fileURLToPath(import.meta.url));
const root = join(homedir(), 'Library/Caches/ms-playwright');
const dir = readdirSync(root).find(d => d.startsWith('chromium_headless_shell'));
if (!dir) throw new Error('chrome-headless-shell не найден в ms-playwright');
const exe = join(root, dir, 'chrome-headless-shell-mac-arm64', 'chrome-headless-shell');
if (!existsSync(exe)) throw new Error('chrome-headless-shell exe не найден: ' + exe);

const browser = await chromium.launch({ executablePath: exe });
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
