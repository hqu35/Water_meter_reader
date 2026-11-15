'use strict';
const assert=require('node:assert/strict');
const G=require('./geometry.js');
const close=(a,b)=>assert.ok(Math.abs(a-b)<1e-8,`${a} != ${b}`);
const register=[[10,10],[8,60],[190,70],[180,20]];
const mix=(a,b,t)=>a.map((v,i)=>v+t*(b[i]-v));
const rails=G.rails(register,220,100);
assert.equal(rails.valid,true);
for(const n of [5,6]) {
 const points=[...register.map(p=>p.slice()),mix(register[1],register[2],.13),mix(register[0],register[3],.12)];
 const result=G.generate(points,n,220,100);
 assert.equal(result.valid,true,result.errors.join('; '));assert.equal(result.mode,'automatic');
 assert.equal(result.expected_points,6);assert.equal(result.quadrilaterals.length,n);
 const first=[register[0],register[1],points[4],points[5]];
 assert.deepEqual(result.quadrilaterals[0],first);
 assert.deepEqual(result.quadrilaterals.at(-1).slice(2),[register[2],register[3]]);
 for(const [name,left,right] of [['top',0,3],['bottom',1,2]]) {
  for(let i=0;i<n;i++) {
   const q=result.quadrilaterals[i];close(Math.hypot(q[right][0]-q[left][0],q[right][1]-q[left][1]),result.spacing[name].cell_width_px);
   if(i) {
    const previous=result.quadrilaterals[i-1][right];
    close(Math.hypot(q[left][0]-previous[0],q[left][1]-previous[1]),result.spacing[name].gap_px);
   }
  }
 }
 // Point seven immediately discards predictions. Moving/previewing alone does not.
 const hover=G.candidate(points,[70,37],n,220,100);assert.equal(hover.starts_manual_fallback,true);
 assert.equal(G.generate(points,n,220,100).quadrilaterals.length,n);
 const manual=[...points,mix(register[0],register[3],.24)];
 const switched=G.generate(manual,n,220,100);
 assert.equal(switched.mode,'manual_fallback');assert.equal(switched.valid,false);
 assert.equal(switched.spacing,null);assert.equal(switched.expected_points,4*n+2);
 assert.deepEqual(switched.quadrilaterals,[first]);
 assert.deepEqual(G.generate(G.undo(manual),n,220,100).quadrilaterals,result.quadrilaterals);
 // Deliberately unequal widths and gaps: fallback must retain manual coordinates.
 manual.push(mix(register[1],register[2],.23),mix(register[1],register[2],.36),mix(register[0],register[3],.38));
 for(let i=2;i<n;i++) {
  const left=.41+(i-2)*.12,right=left+.08+(i%2)*.01;
  manual.push(mix(register[0],register[3],left),mix(register[1],register[2],left),
    mix(register[1],register[2],right),mix(register[0],register[3],right));
 }
 const completed=G.generate(manual,n,220,100);
 assert.equal(manual.length,4*n+2);assert.equal(completed.valid,true,completed.errors.join());
 assert.equal(completed.quadrilaterals.length,n);assert.equal(completed.mode,'manual_fallback');
 assert.deepEqual(completed.quadrilaterals.slice(1).flat(),manual.slice(6));
 assert.notDeepEqual(completed.quadrilaterals.at(-1).slice(2),[register[2],register[3]]);
 assert.equal(G.candidate(manual,[50,40],n,220,100),null);
 assert.equal(G.generate(G.undo(manual),n,220,100).quadrilaterals.length,n-1);
 const damaged=manual.map(p=>p.slice());damaged[6][1]+=2;
 assert.match(G.generate(damaged,n,220,100).errors.join(),/not on its register rail/);
 const crossed=manual.map(p=>p.slice());[crossed[8],crossed[9]]=[crossed[9],crossed[8]];
 assert.equal(G.generate(crossed,n,220,100).valid,false);
 const tooWide=points.map(p=>p.slice());tooWide[5]=mix(register[0],register[3],.3);
 const invalid=G.generate(tooWide,n,220,100);assert.equal(invalid.valid,false);assert.equal(invalid.quadrilaterals.length,0);
 assert.match(invalid.errors.join(),/fit/);
 // A first cell too wide for equal spacing must still allow manual fallback.
 const wideManual=G.generate([...tooWide,mix(register[0],register[3],.4)],n,220,100);
 assert.equal(wideManual.mode,'manual_fallback');assert.equal(wideManual.quadrilaterals.length,1);
 assert.ok(!wideManual.errors.some(e=>e.startsWith('Spacing:')));
 const zeroGap=points.map(p=>p.slice());zeroGap[4]=mix(register[1],register[2],1/n);zeroGap[5]=mix(register[0],register[3],1/n);
 const touching=G.generate(zeroGap,n,220,100);assert.equal(touching.valid,true,touching.errors.join());
 close(touching.spacing.top.gap_px,0);close(touching.spacing.bottom.gap_px,0);
}
const partial=register.map(p=>p.slice());
for(const [role,rail] of [['BR','bottom'],['TR','top'],['TL','top'],['BL','bottom'],['BR','bottom'],['TR','top'],['TL','top']]) {
 const preview=G.candidate(partial,[70,37],5,220,100);
 assert.equal(preview.rail,rail);assert.equal(preview.role,role);
 const [a,b,c]=rails[rail].equation;close(a*preview.point[0]+b*preview.point[1]+c,0);
 partial.push(preview.point);
}
const free=G.candidate(register.slice(0,3),[180,20],5,220,100);
assert.equal(free.stage,'register');assert.deepEqual(free.point,[180,20]);
assert.equal(G.rails([register[0],register[2],register[1],register[3]],220,100).valid,false);
assert.equal(G.candidate([[1,1],[1,1],[2,2],[2,1]],[50,40],5,220,100),null);
assert.equal(G.projectToRail([5,5],{start:[1,1],end:[1,1]}),null);
const left=G.projectToRail([-100,0],rails.top);assert.equal(left.clamped,true);assert.deepEqual(left.point,register[0]);
const p=G.projectToRail([65,45],rails.top,[3,.5]);
const d=[rails.top.end[0]-rails.top.start[0],rails.top.end[1]-rails.top.start[1]];
close((65-p.point[0])*d[0]*9+(45-p.point[1])*d[1]*.25,0);
for(const [w,h,rect] of [[99,30,{left:34,top:18,width:1188,height:360}],[425,131,{left:100,top:60,width:850,height:262}],[308,82,{left:7,top:19,width:451,height:120}]]) {
 for(const [fx,fy] of [[0,0],[.5,.5],[.999,.999]]) {
  const x=rect.left+fx*rect.width,y=rect.top+fy*rect.height;
  const p=G.displayToImage(x,y,rect,w,h);
  close(rect.left+p[0]*rect.width/w,x);close(rect.top+p[1]*rect.height/h,y);
 }
}
assert.equal(G.displayToImage(0,0,{left:10,top:20,width:100,height:50},201,101),null);
console.log('Guided geometry: rail projection, roles, 6-point auto, point-7 fallback, 22/26-point manual completion, validation, undo and screen-coordinate round trips passed.');

// The live guide follows the side shear, rather than an edge normal.
const parallel=[[30,10],[10,70],[190,70],[210,10]];
const tilted=G.rails(parallel,240,100);
close(tilted.side_direction.mean_dx_dy,-1/3);
let guide=G.candidate(parallel,[100,40],5,240,100);
close(guide.point[0],90);close(guide.point[1],70);
const afterTop=[...parallel,guide.point];
guide=G.candidate(afterTop,[100,40],5,240,100);
close(guide.point[0],110);close(guide.point[1],10);
const unequal=[[30,10],[10,70],[180,70],[210,10]];
const mean=G.meanSideDirection(unequal);close(mean.mean_dx_dy,(-1/3-.5)/2);
guide=G.candidate(unequal,[100,40],5,240,100);close(guide.point[0],87.5);
const vertical=[[10,10],[10,70],[210,70],[210,10]];
assert.deepEqual(G.meanSideDirection(vertical).vector,[0,1]);
assert.deepEqual(G.candidate(vertical,[100,40],5,240,100).point,[100,70]);
const oneVertical=[[10,10],[10,70],[180,70],[210,10]];
close(G.meanSideDirection(oneVertical).mean_dx_dy,-.25);
for(const q of [register,parallel,unequal,oneVertical]) {
 const v=G.meanSideDirection(q).vector;
 const candidate=G.candidate(q,[100,40],5,240,100);
 const delta=[candidate.point[0]-100,candidate.point[1]-40];
 close(delta[0]*v[1]-delta[1]*v[0],0);
 // The same parallelism holds after a nonuniform CSS scale.
 close((delta[0]*3)*(v[1]*.6)-(delta[1]*.6)*(v[0]*3),0);
}
assert.equal(G.candidate(parallel,[1,40],5,240,100),null);
assert.equal(G.projectAlongDirection([1,2],{start:[0,0],end:[10,0]},[1,0]),null);
console.log('Oblique guide: parallel sides, averaged side shear, vertical sides, screen scaling and out-of-segment rejection passed.');
