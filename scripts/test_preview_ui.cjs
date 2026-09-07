// Synthetic UI fixture ONLY; never a benchmark or publishable recording.
const {chromium}=require('playwright');
const http=require('node:http'),fs=require('node:fs'),assert=require('node:assert/strict');
(async()=>{
 let started=false;
 const server=http.createServer((req,res)=>{
  if(req.method==='POST'){started=true;res.writeHead(202);res.end();return;}
  if(req.url==='/'){res.setHeader('Content-Type','text/html');res.end(fs.readFileSync(__dirname+'/preview_stream.html'));return;}
  const since=Number(new URL(req.url,'http://localhost').searchParams.get('since')||0);
  res.setHeader('Content-Type','application/json');
  res.end(JSON.stringify({label:'EXCLUDED SYNTHETIC UI TEST',status:started?'completed':'ready',events:started&&since===0?Array.from({length:8},(_,index)=>({index,chunk:{text:'<not HTML> real chunk',meta_info:{completion_tokens:2,finish_reason:{type:'stop'}}}})):[]}));
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
  assert.equal(await page.locator('pre').first().textContent(),'<not HTML> real chunk');
  assert.deepEqual(errors,[]);
  console.log('PASS: actual DOM renders all eight chunk streams without HTML interpretation');
 }finally{await browser.close();await new Promise(resolve=>server.close(resolve));}
})().catch(e=>{console.error(e.message);process.exitCode=1;});
