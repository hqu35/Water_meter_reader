/* Browser event harness: no real labels or browser storage are modified. */
'use strict';
const assert=require('node:assert/strict');
const fs=require('node:fs');
const vm=require('node:vm');
const items=[5,6].map((n,i)=>({image_id:'fixture'+i,sha256:'hash'+i,orientation:i===0?'upside_down':'upright',
  digit_count:n,width:200,height:80,six_digit:Number(n===6),src:'src'+i}));
const dataset={images:items,fingerprint:'test-guided',excluded_original_ids:[]};
const prefix='guided-hybrid-cells-v1:test-guided:';
const previousPrefix='guided-equal-spacing-v1:test-guided:';
const legacyKey='manual-all-cells-v1:test-guided:image:fixture0';
const legacyRaw=JSON.stringify({schema:'watermeter-manual-cells-v1',image_id:'fixture0',source_image_sha256:'hash0',manual_points:[[1,1]]});
const storage=new Map([[legacyKey,legacyRaw]]);
let lastBlob, alerts=[],allowReset=true,failWrites=false;
function start() {
 const nodes={},events={},windowEvents={};let guideScreen=null,closedPaths=0;
 const ctx=new Proxy({}, {get:(_,name)=>name==='arc'?(x,y,r)=>{if(r===4.5)guideScreen=[x,y]}:name==='clearRect'?()=>{closedPaths=0}:name==='closePath'?()=>{closedPaths++}:()=>{},set:()=>true});
 const element=id=>nodes[id]||(nodes[id]={style:{},textContent:'',className:'',clientWidth:1020,clientHeight:420,value:'',
   getContext:()=>ctx,getBoundingClientRect:()=>({left:100,top:80,width:1000,height:400}),focus(){},click(){if(this.onclick)this.onclick()}});
 element('dataset').textContent=JSON.stringify(dataset);
 const context={console,Map,Set,Math,Number,Array,Object,JSON,Error,Blob,devicePixelRatio:2,
  requestAnimationFrame:fn=>fn(),setTimeout(){},confirm:()=>allowReset,alert:m=>alerts.push(m),
  URL:{createObjectURL:b=>(lastBlob=b,'blob:test'),revokeObjectURL(){}},
  localStorage:{getItem:k=>storage.get(k)??null,setItem:(k,v)=>{if(failWrites)throw Error('quota');storage.set(k,v)},removeItem:k=>storage.delete(k)},
  document:{getElementById:element,addEventListener:(name,fn)=>events[name]=fn,body:{focus(){},appendChild(){}},createElement:()=>({click(){},remove(){}})},
  Image:class {set src(value){this.naturalWidth=200;this.naturalHeight=80;this.onload()}}};
 context.window=context;context.addEventListener=(name,fn)=>windowEvents[name]=fn;
 vm.createContext(context);vm.runInContext(fs.readFileSync('geometry.js','utf8'),context);
 vm.runInContext(fs.readFileSync('guided_app.js','utf8'),context);
 const key=(code,key)=>events.keydown({code,key,target:{closest:()=>false},preventDefault(){},shiftKey:false});
 function mark(x,y,id) {
  events.pointermove({clientX:100+x*5,clientY:80+y*5});
  const displayed=guideScreen.slice();key('Space',' ');
  const record=JSON.parse(storage.get(prefix+'image:'+id));const saved=record.manual_points.at(-1);
  assert.ok(Math.abs(displayed[0]-saved[0]*5)<1e-8);
  assert.ok(Math.abs(displayed[1]-saved[1]*5)<1e-8);
 }
 return {nodes,events,key,mark,closedPaths:()=>closedPaths};
}
(async()=>{
 const G=require('./geometry.js');
 const outline=[[5,5],[7,65],[190,70],[185,15]];
 const base=outline.map(p=>p.slice());
 for(let i=0;i<2;i++)base.push(G.candidate(base,[30,35],5,200,80).point);
 const previousRecords=items.map((item,i)=>({schema:'watermeter-guided-equal-spacing-v1',
  annotation_tool:'register_rail_equal_spacing_browser',image_id:item.image_id,source_image_sha256:item.sha256,
  orientation_label:item.orientation,digit_count:item.digit_count,
  manual_points:i?[...base,G.candidate(base,[42,35],item.digit_count,200,80).point,
    G.candidate([...base,[42,8]],[42,35],item.digit_count,200,80).point]:base,
  status:i?'complete':'in_progress'}));
 const previousBytes=previousRecords.map(r=>JSON.stringify(r));
 items.forEach((item,i)=>storage.set(previousPrefix+'image:'+item.image_id,previousBytes[i]));
 let ui=start();
 assert.match(ui.nodes.progress.textContent,/Complete 2/);
 assert.match(ui.nodes.step.textContent,/AUTO/);
 assert.equal(storage.has(prefix+'image:fixture0'),false,'Read-only restore must not rewrite old storage');
 for(let index=0;index<2;index++) {
  const item=items[index];ui.nodes.reset.onclick();
  for(const p of outline)ui.mark(...p,item.image_id);
  for(const p of [[30,35],[30,35]])ui.mark(...p,item.image_id);
  let record=JSON.parse(storage.get(prefix+'image:'+item.image_id));
  assert.equal(record.status,'complete');assert.equal(record.manual_points.length,6);
  assert.equal(record.annotation_mode,'automatic');assert.equal(record.digit_quadrilaterals.length,item.digit_count);
  assert.equal(ui.closedPaths(),1+item.digit_count);
  // Mouse movement alone must not switch modes or remove proposals.
  ui.events.pointermove({clientX:100+40*5,clientY:80+35*5});
  assert.equal(JSON.parse(storage.get(prefix+'image:'+item.image_id)).annotation_mode,'automatic');
  assert.equal(ui.closedPaths(),1+item.digit_count);
  ui.mark(40,35,item.image_id);
  record=JSON.parse(storage.get(prefix+'image:'+item.image_id));
  assert.equal(record.annotation_mode,'manual_fallback');assert.equal(record.status,'in_progress');
  assert.equal(record.manual_points.length,7);assert.equal(record.digit_quadrilaterals.length,1);
  assert.equal(record.spacing,null);assert.equal(ui.closedPaths(),2,'Only outer register and first cell may remain');
  assert.match(ui.nodes.step.textContent,/MANUAL digit 2.*BL/);
  // Undo across the transition restores the automatic proposals.
  ui.key('KeyC','c');assert.equal(ui.closedPaths(),1+item.digit_count);
  assert.equal(JSON.parse(storage.get(prefix+'image:'+item.image_id)).annotation_mode,'automatic');
  ui.mark(40,35,item.image_id);
  // Resume a partial fallback without exposing any stale predicted polygons.
  ui=start();assert.equal(ui.closedPaths(),2);assert.match(ui.nodes.step.textContent,/MANUAL digit 2/);
  for(let digit=1;digit<item.digit_count;digit++) {
   const x=40+(digit-1)*27;
   const corners=[[x,35],[x,35],[x+20,35],[x+20,35]];
   for(const point of digit===1?corners.slice(1):corners)ui.mark(...point,item.image_id);
  }
  record=JSON.parse(storage.get(prefix+'image:'+item.image_id));
  assert.equal(record.status,'complete');assert.equal(record.manual_points.length,4*item.digit_count+2);
  assert.equal(record.expected_point_count,record.manual_points.length);
  assert.equal(record.digit_quadrilaterals.length,item.digit_count);
  assert.equal(record.digit_points.length,4*item.digit_count);
  assert.deepEqual(record.digit_quadrilaterals[0].slice(0,2),record.register_corners.slice(0,2));
  assert.deepEqual(record.digit_quadrilaterals.slice(1).flat(),record.manual_points.slice(6));
  assert.equal(record.geometry.equal_width_per_rail,false);assert.equal(record.geometry.equal_gap_per_rail,false);
  ui.key('Space',' ');assert.equal(JSON.parse(storage.get(prefix+'image:'+item.image_id)).manual_points.length,record.manual_points.length);
  ui.key('KeyC','c');assert.equal(JSON.parse(storage.get(prefix+'image:'+item.image_id)).status,'in_progress');
  const last=record.manual_points.at(-1);ui.mark(last[0],35,item.image_id);
  if(index===0)ui.key('Period','。');
 }
 ui.key('Comma','，');assert.match(ui.nodes['image-meta'].textContent,/fixture0/);
 ui.nodes.export.onclick();const payload=JSON.parse(await lastBlob.text());
 assert.equal(payload.schema,'watermeter-guided-hybrid-cells-export-v1');assert.equal(payload.annotations.length,2);
 assert.equal(storage.get(legacyKey),legacyRaw);
 items.forEach((item,i)=>assert.equal(storage.get(previousPrefix+'image:'+item.image_id),previousBytes[i]));
 ui.nodes['legacy-export'].onclick();const oldBackup=JSON.parse(await lastBlob.text());
 assert.equal(oldBackup.schema,'watermeter-manual-cells-export-v1');
 assert.deepEqual(oldBackup.annotations,[JSON.parse(legacyRaw)]);
 ui=start();assert.match(ui.nodes.progress.textContent,/Complete 2/);
 const importFile=async value=>ui.nodes['import-file'].onchange({target:{files:[{text:async()=>JSON.stringify(value)}],value:'file'}});
 await importFile(payload);assert.equal(alerts.length,0);
 const damaged=structuredClone(payload);damaged.annotations[0].manual_points[6][1]+=3;
 await importFile(damaged);assert.match(alerts.pop(),/off-rail/);
 await importFile({...payload,schema:'watermeter-manual-cells-export-v1'});assert.match(alerts.pop(),/Older annotation formats/);
 // Import old 6/8-point records without turning diagnostic points into manual boxes.
 await importFile({...payload,schema:'watermeter-guided-equal-spacing-export-v1',annotations:previousRecords});
 assert.equal(alerts.length,0);
 for(const item of items) {
  const restored=JSON.parse(storage.get(prefix+'image:'+item.image_id));
  assert.equal(restored.manual_points.length,6);assert.equal(restored.annotation_mode,'automatic');
  assert.deepEqual(restored.previous_equal_spacing_record,previousRecords[items.indexOf(item)]);
  assert.equal(restored.digit_quadrilaterals.length,item.digit_count);
 }
 await importFile(payload);assert.equal(alerts.length,0);
 allowReset=false;ui.nodes.reset.onclick();assert.match(ui.nodes.step.textContent,/COMPLETE/);
 allowReset=true;ui.nodes.reset.onclick();assert.equal(JSON.parse(storage.get(prefix+'image:fixture0')).manual_points.length,0);
 assert.match(ui.nodes.step.textContent,/Register outline/);
 for(const p of outline)ui.mark(...p,'fixture0');
 ui.key('KeyC','c');assert.match(ui.nodes.step.textContent,/TR/);
 failWrites=true;ui.events.pointermove({clientX:600,clientY:180});ui.key('Space',' ');
 assert.match(ui.nodes.warning.textContent,/Could not save/);
 assert.equal(JSON.parse(storage.get(prefix+'image:fixture0')).manual_points.length,3);
 items.forEach((item,i)=>assert.equal(storage.get(previousPrefix+'image:'+item.image_id),previousBytes[i]));
 console.log('GUI passed: 6-point auto; point-7 removal of predicted overlays; 22/26 manual points; cursor equality; undo/resume; import/export; old 6/8-point record preservation; storage failure.');
})().catch(e=>{console.error(e);process.exitCode=1});
