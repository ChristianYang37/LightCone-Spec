// Synthetic UI fixture ONLY; never a benchmark or publishable recording.
const {chromium}=require('playwright');
const http=require('node:http'),fs=require('node:fs'),assert=require('node:assert/strict');
(async()=>{
 let started=false, clients=[],submitted=0,requestCount=8;
 const common={label:'EXCLUDED SYNTHETIC UI TEST',model:'SYNTHETIC',tp:2,dispatcher_concurrency:8,input_tokens_per_request:16384,max_output_tokens:1024};
 const send=(res,type,data,id)=>res.write(`${id?`id: ${id}\n`:''}event: ${type}\ndata: ${JSON.stringify(data)}\n\n`);
 const complete=res=>{
  for(let index=0;index<requestCount;index++){
   send(res,'token',{sequence:2*index+1,index,elapsed_seconds:.5,token_ids:[1],chunk:{output_ids:[1],text:'中�',meta_info:{completion_tokens:1,finish_reason:null}}},2*index+1);
   send(res,'token',{sequence:2*index+2,index,elapsed_seconds:1,token_ids:[2],chunk:{output_ids:[1,2],text:'中文🙂 <not HTML>',meta_info:{completion_tokens:2,finish_reason:{type:'stop'}}}},2*index+2);
  }
  send(res,'state',{...common,status:'completed',event_count:2*requestCount,committed_tokens:2*requestCount,duration_seconds:2,aggregate_tok_s:requestCount,per_user_tok_s:12});res.end();
 };
 const server=http.createServer((req,res)=>{
  if(req.url==='/clock'){res.setHeader('Content-Type','application/json');res.end(JSON.stringify({recording_hostname:require('node:os').hostname(),server_epoch_ns:String(BigInt(Date.now())*1000000n)}));return;}
  if(req.url==='/status'){res.setHeader('Content-Type','application/json');res.end(JSON.stringify({...common,status:'completed',submitted_at_ns:submitted,event_count:16,committed_tokens:16,duration_seconds:2,aggregate_tok_s:8,per_user_tok_s:12}));return;}
  if(req.method==='POST'){started=true;submitted=Date.now()*1e6;res.writeHead(202);res.end();setTimeout(()=>{clients.forEach(complete);clients=[];},500);return;}
  if(req.url==='/'){res.setHeader('Content-Type','text/html');res.end(fs.readFileSync(__dirname+'/preview_stream.html'));return;}
  res.setHeader('Content-Type','text/event-stream');res.flushHeaders();
  if(started){complete(res);return;}
  send(res,'state',{...common,status:'ready'});clients.push(res);
 });
 await new Promise(resolve=>server.listen(0,'127.0.0.1',resolve));
 server.unref();
 const browser=await chromium.launch({headless:true});
 try{
  const page=await browser.newPage();const errors=[];page.on('pageerror',e=>errors.push(e.message));
  await page.goto(`http://127.0.0.1:${server.address().port}`);
  await page.waitForFunction(()=>document.body.dataset.status==='ready');
  await page.click('#start');
  await page.waitForFunction(()=>document.body.dataset.status==='completed');
  assert.equal(await page.locator('article').count(),8);
  assert.match(await page.locator('#metrics').textContent(),/16 committed tokens/);
  assert.equal(await page.locator('pre').first().textContent(),'中文🙂 <not HTML>');
  assert.match(await page.locator('#metrics').textContent(),/MEASURED end-to-end throughput: 8.0 tok\/s/);
  assert.match(await page.locator('#configuration').textContent(),/TP2/);
  await page.evaluate(()=>guard(receive)({lastEventId:'18',data:JSON.stringify({sequence:18})}));
  assert.equal(await page.locator('body').getAttribute('data-status'),'failed');
  assert.match(await page.locator('#metrics').textContent(),/Lost or duplicate/);
  await page.goto('file://'+__dirname+'/preview_stream.html');
  await page.waitForFunction(()=>document.body.dataset.status==='failed');
  assert.match(await page.locator('#metrics').textContent(),/local HTML file/);
  assert.deepEqual(errors,[]);
  started=false;clients=[];requestCount=48;common.request_count=48;
  common.video_disclosure={state_scope:'cohort'};
  await page.goto(`http://127.0.0.1:${server.address().port}`);
  await page.waitForFunction(()=>document.body.dataset.status==='ready');
  await page.click('#start');await page.waitForFunction(()=>document.body.dataset.status==='completed');
  assert.equal(await page.locator('article').count(),48);
  assert.equal(await page.locator('article:visible').count(),8);
  assert.match(await page.locator('#metrics').textContent(),/48\/48 completed/);
  assert.match(await page.locator('#configuration').textContent(),/SELECTED EFFECT CASE, NOT AVERAGE PERFORMANCE/);
  requestCount=8;delete common.request_count;delete common.video_disclosure;
  if(process.env.PREVIEW_CAPTURE_TEST_DIRECTORY){
   // Optional real frame/codec smoke against this explicitly synthetic fixture;
   // these artifacts must never be presented as a model benchmark/video.
   started=false;clients=[];
   const child=require('node:child_process').spawn(process.execPath,[__dirname+'/capture_preview.cjs',
    `http://127.0.0.1:${server.address().port}`,process.env.PREVIEW_CAPTURE_TEST_DIRECTORY],{stdio:'inherit'});
   const code=await new Promise(resolve=>child.on('exit',resolve));
   assert.equal(code,0,'real CDP frame capture and encoding');
   const capture=JSON.parse(fs.readFileSync(process.env.PREVIEW_CAPTURE_TEST_DIRECTORY+'/capture.json'));
   assert.equal(capture.alignment.status,'verified_clock_bounded');
   assert.ok(capture.frame_count>1);
   assert.equal(capture.measurement.model,'SYNTHETIC');
  }
  console.log('PASS: actual DOM renders all eight chunk streams without HTML interpretation');
 }finally{await browser.close();await new Promise(resolve=>server.close(resolve));}
})().catch(e=>{console.error(e.message);process.exitCode=1;});
