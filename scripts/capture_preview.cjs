// New contributions: LicenseRef-LightCone-Source-Available-1.0. See LICENSE.
// Capture actual browser frames with native wall timestamps, not a token replay.
const {chromium}=require('playwright');
const fs=require('node:fs'),path=require('node:path'),os=require('node:os'),{spawnSync}=require('node:child_process'),{createHash}=require('node:crypto');
async function clockBound(url){
 const samples=[];
 for(let i=0;i<8;i++){
  const before=Date.now()/1000,response=await fetch(new URL('/clock',url));
  if(!response.ok)throw Error('Recording service clock unavailable');
  const data=await response.json(),after=Date.now()/1000,server=Number(data.server_epoch_ns)/1e9;
  if(data.recording_hostname!==os.hostname())throw Error('Capture must run on the SSH server itself, not through a network tunnel');
  if(!Number.isFinite(server))throw Error('Invalid service clock');
  samples.push({lower:server-after-.001,upper:server-before+.001,rtt:after-before});
 }
 return samples.sort((a,b)=>a.rtt-b.rtt)[0];
}
(async()=>{
 const [url,out]=process.argv.slice(2);
 if(!url||!out)throw Error('Usage: node capture_preview.cjs http://127.0.0.1:PORT OUTPUT_DIRECTORY');
 if(!['127.0.0.1','localhost'].includes(new URL(url).hostname))throw Error('Use the recording server local loopback URL');
 fs.mkdirSync(out,{recursive:false});fs.mkdirSync(path.join(out,'frames'));
 const browser=await chromium.launch({headless:true});
 const context=await browser.newContext({viewport:{width:1600,height:1050}});
 const page=await context.newPage(),cdp=await context.newCDPSession(page),errors=[],frames=[];
 page.on('pageerror',e=>errors.push(e.message));
 cdp.on('Page.screencastFrame',event=>{
  try{
   const timestamp=event.metadata.timestamp;
   if(!Number.isFinite(timestamp)||timestamp<1e9||
      (frames.length&&timestamp<=frames.at(-1).timestamp))throw Error('Invalid native frame timestamps');
   const filename=`frames/frame-${String(frames.length).padStart(7,'0')}.png`;
   fs.writeFileSync(path.join(out,filename),Buffer.from(event.data,'base64'));
   frames.push({filename,timestamp});
  }catch(error){errors.push(error.message);}
  cdp.send('Page.screencastFrameAck',{sessionId:event.sessionId}).catch(e=>errors.push(e.message));
 });
 let status='failed',measurement=null,alignment=null,clockBefore=null,clockAfter=null;
 try{
  await cdp.send('Page.enable');
  await cdp.send('Page.startScreencast',{format:'png',everyNthFrame:1,maxWidth:1600,maxHeight:1050});
  await page.goto(url); await page.waitForFunction(()=>document.body.dataset.status==='ready');
  clockBefore=await clockBound(url);
  await page.screenshot({path:path.join(out,'before.png')});
  await page.click('#start');
  await page.waitForFunction(()=>['completed','failed'].includes(document.body.dataset.status),{},{timeout:3600000});
  status=await page.locator('body').getAttribute('data-status');
  measurement=await (await fetch(new URL('/status',url))).json();
  clockAfter=await clockBound(url);
  await page.screenshot({path:path.join(out,'after.png')});
  await cdp.send('Page.stopScreencast');
  if(status!=='completed'||measurement.status!=='completed'||errors.length||frames.length<2)
   throw Error('Capture failed; preserve this take');
  const lower=Math.min(clockBefore.lower,clockAfter.lower),upper=Math.max(clockBefore.upper,clockAfter.upper);
  const submitted=Number(measurement.submitted_at_ns)/1e9;
  const offset=submitted-(lower+upper)/2-frames[0].timestamp,uncertainty=(upper-lower)/2;
  if(!Number.isFinite(offset)||offset<uncertainty||uncertainty>.1||
     frames.at(-1).timestamp<submitted-lower)throw Error('Submission alignment clock bound invalid or exceeds 100 ms');
  alignment={status:'verified_clock_bounded',submission_offset_seconds:offset,
   clock_uncertainty_seconds:uncertainty,clock_before:clockBefore,clock_after:clockAfter,
   origin:'CDP screencast native first-frame epoch; service submission timestamp',
   display:'browser-arrival frames; no fabricated per-token timing'};
 }catch(error){status='failed';errors.push(error.message);}
 finally{
  await context.close();await browser.close();
  fs.writeFileSync(path.join(out,'frames.json'),JSON.stringify(frames,null,2));
  if(frames.length>1){
   // Retain observed intervals; unchanged frames are held, never interpolated.
   const concat=['ffconcat version 1.0'];
   frames.forEach((frame,i)=>{concat.push(`file '${frame.filename}'`,'option framerate 1000');if(i+1<frames.length)concat.push(`duration ${(frames[i+1].timestamp-frame.timestamp).toFixed(9)}`);});
   fs.writeFileSync(path.join(out,'frames.ffconcat'),concat.join('\n')+'\n');
   // This concat file is generated above from internal integer-only filenames;
   // safe=0 permits the explicit millisecond PNG time base, not user paths.
   const encoded=spawnSync(process.env.FFMPEG||'ffmpeg',['-n','-f','concat','-safe','0','-i','frames.ffconcat','-fps_mode','vfr','-c:v','libx264','-crf','18','-pix_fmt','yuv420p','original.mp4'],{cwd:out,encoding:'utf8'});
   fs.writeFileSync(path.join(out,'encode.log'),encoded.stderr||String(encoded.error||''));
   if(encoded.status!==0){status='failed';errors.push('Video encoding failed; original timestamped PNG frames retained');}
  }
  const original=path.join(out,'original.mp4');
  const video_sha256=fs.existsSync(original)?createHash('sha256').update(fs.readFileSync(original)).digest('hex'):null;
  fs.writeFileSync(path.join(out,'capture.json'),JSON.stringify({status,errors,playback_speed:1,video_sha256,
   mode:'actual_live_browser_capture',measurement,alignment,frame_count:frames.length},null,2));
 }
 if(status!=='completed')throw Error(errors.join('; '));
})().catch(e=>{console.error(e.message);process.exitCode=1;});
