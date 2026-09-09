// Synthetic rendering test only. No GPU result or downloaded model is used.
const {chromium} = require('playwright');
const {mkdtempSync} = require('node:fs');
const {tmpdir} = require('node:os');
const {join} = require('node:path');
const {execFileSync} = require('node:child_process');
const assert = require('node:assert/strict');
const output = mkdtempSync(join(tmpdir(), 'context-benchmark-synthetic-'));
execFileSync(process.env.PYTHON || 'python', ['-c', `
from lightcone_spec.preview_benchmark import write_report, FIXED_VERSION, FIXED_GATE, ADAPTIVE_VERSION
write_report(${JSON.stringify(output)}, [], {'scope':'SYNTHETIC UI test; not measured'})
write_report(${JSON.stringify(join(output,'fixed'))}, [], {'benchmark_version':FIXED_VERSION,'fixed_gate':FIXED_GATE})
write_report(${JSON.stringify(join(output,'adaptive'))}, [], {'benchmark_version':ADAPTIVE_VERSION,'fixed_gate':FIXED_GATE})
`]);
(async()=>{
 const browser = await chromium.launch({headless:true});
 try {
  const page = await browser.newPage();
  const errors=[];page.on('pageerror', error=>errors.push(String(error)));
  await page.goto('file://' + join(output,'index.html'));
  assert.equal(await page.locator('#table tbody td').count(),30);
  assert.equal(await page.locator('#delta tbody td').count(),30);
  await page.selectOption('#mode','gated_s10');await page.selectOption('#metric','al');
  assert.match(await page.locator('#score').innerText(),/UNMEASURED/);
  assert.deepEqual(errors,[]);
  await page.goto('file://' + join(output,'fixed','index.html'));
  assert.equal(await page.locator('#mode option').count(),4);
  await page.selectOption('#mode','gated_s5');
  assert.equal(await page.locator('#table tbody td').count(),30);
  assert.match(await page.locator('#score').innerText(),/UNMEASURED/);
  assert.deepEqual(errors,[]);
  await page.goto('file://' + join(output,'adaptive','index.html'));
  assert.deepEqual(await page.locator('#mode option').allTextContents(),['always_s10','always_s64','gated_s10','gated_s5']);
  await page.selectOption('#mode','always_s64');
  assert.equal(await page.locator('#table tbody td').count(),30);
  assert.match(await page.locator('#comparison').innerText(),/always_s10/);
  assert.deepEqual(errors,[]);
  console.log('PASS: synthetic 3x10 tables, method/metric switches, no fabricated scores');
 } finally {await browser.close();}
})().catch(error=>{console.error(error);process.exitCode=1;});
