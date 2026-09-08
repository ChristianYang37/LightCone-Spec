// Synthetic rendering test only. No GPU result or downloaded model is used.
const {chromium} = require('playwright');
const {mkdtempSync} = require('node:fs');
const {tmpdir} = require('node:os');
const {join} = require('node:path');
const {execFileSync} = require('node:child_process');
const assert = require('node:assert/strict');
const output = mkdtempSync(join(tmpdir(), 'context-benchmark-synthetic-'));
execFileSync(process.env.PYTHON || 'python', ['-c', `
from lightcone_spec.preview_benchmark import write_report
write_report(${JSON.stringify(output)}, [], {'scope':'SYNTHETIC UI test; not measured'})
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
  console.log('PASS: synthetic 3x10 tables, method/metric switches, no fabricated scores');
 } finally {await browser.close();}
})().catch(error=>{console.error(error);process.exitCode=1;});
