// New contributions: LicenseRef-LightCone-Source-Available-1.0. See LICENSE.
// Record an actual browser session; does NOT replay stored token events.
const {chromium}=require('playwright');
const fs=require('node:fs');
const path=require('node:path');
(async()=>{
 const [url,out]=process.argv.slice(2);
 if(!url||!out)throw Error('Usage: node capture_preview.cjs http://127.0.0.1:PORT OUTPUT_DIRECTORY');
 if(!['127.0.0.1','localhost'].includes(new URL(url).hostname))throw Error('Use a loopback SSH tunnel');
 fs.mkdirSync(out,{recursive:false});
 const browser=await chromium.launch({headless:true});
 const context=await browser.newContext({viewport:{width:1600,height:1050},recordVideo:{dir:out,size:{width:1600,height:1050}}});
 const page=await context.newPage(); const errors=[];page.on('pageerror',e=>errors.push(e.message));
 try{
  await page.goto(url); await page.waitForFunction(()=>document.body.dataset.status==='ready');
  await page.screenshot({path:path.join(out,'before.png')});
  await page.click('#start');
  await page.waitForFunction(()=>['completed','failed'].includes(document.body.dataset.status),{},{timeout:3600000});
  const status=await page.locator('body').getAttribute('data-status');
  await page.screenshot({path:path.join(out,'after.png')});
  fs.writeFileSync(path.join(out,'capture.json'),JSON.stringify({status,errors,playback_speed:1,mode:'actual_live_browser_capture'},null,2));
  if(status!=='completed'||errors.length)throw Error('Capture failed; preserve this take');
 }finally{await context.close();await browser.close();}
})().catch(e=>{console.error(e.message);process.exitCode=1;});
