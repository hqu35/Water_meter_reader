/* Original-resolution coordinates. Only digit points are projected onto rails. */
(function (root, factory) {
  const api = factory();
  if (typeof module === 'object' && module.exports) module.exports = api;
  if (root) root.GuidedRegisterGeometry = api;
})(typeof window === 'undefined' ? null : window, function () {
  'use strict';
  const roles = ['TL', 'BL', 'BR', 'TR'];
  const roleAt = i => i<4?roles[i]:i<6?roles[i-2]:roles[(i-6)%4];
  const manualTotal = count => 6+4*(count-1);
  const finitePoint = p => Array.isArray(p) && p.length === 2 && p.every(Number.isFinite);
  const cross = (a,b,c) => (b[0]-a[0])*(c[1]-a[1]) - (b[1]-a[1])*(c[0]-a[0]);
  const inside = (p,w,h) => finitePoint(p) && p[0]>=0 && p[0]<w && p[1]>=0 && p[1]<h;

  function displayToImage(x,y,rect,width,height) {
    if (!(rect.width>0 && rect.height>0)) return null;
    const dx=x-rect.left, dy=y-rect.top;
    if (!(dx>=0 && dx<rect.width && dy>=0 && dy<rect.height)) return null;
    return [dx*width/rect.width, dy*height/rect.height];
  }

  function validateQuad(q, width, height) {
    const errors=[];
    if (!Array.isArray(q) || q.length!==4 || !q.every(finitePoint)) return ['Four finite corners are required.'];
    if (!q.every(p=>inside(p,width,height))) errors.push('A corner lies outside the image.');
    if (!q.every((_,i)=>cross(q[i],q[(i+1)%4],q[(i+2)%4]) < -1e-7)) errors.push('Corners must form a convex cell ordered TL, BL, BR, TR.');
    const [tl,bl,br,tr]=q;
    if (tl[0]+bl[0]>=tr[0]+br[0] || tl[1]+tr[1]>=bl[1]+br[1]) errors.push('Corners do not match top/bottom and left/right order.');
    return errors;
  }

  function line(start,end) {
    const dx=end[0]-start[0], dy=end[1]-start[1], length=Math.hypot(dx,dy);
    if (!(length>1e-7)) throw Error('Register rail is degenerate.');
    // A*x + B*y + C = 0, normalized to unit normal length.
    return {start:start.slice(),end:end.slice(),equation:[-dy/length,dx/length,(dy*start[0]-dx*start[1])/length]};
  }

  function meanSideDirection(register) {
    if (!Array.isArray(register) || register.length!==4 || !register.every(finitePoint)) return null;
    const [tl,bl,br,tr]=register;
    const leftDy=bl[1]-tl[1], rightDy=br[1]-tr[1];
    if (!(leftDy>1e-7 && rightDy>1e-7)) return null;
    // x(y) slope is finite for vertical sides. Average the two side shears.
    const left=(bl[0]-tl[0])/leftDy, right=(br[0]-tr[0])/rightDy;
    return {vector:[(left+right)/2,1],left_dx_dy:left,right_dx_dy:right,
      mean_dx_dy:(left+right)/2,definition:'mean of side dx/dy; downward direction (mean_dx_dy, 1)'};
  }

  function projectAlongDirection(point,rail,direction) {
    if (!finitePoint(point) || !rail || !finitePoint(rail.start) || !finitePoint(rail.end) || !finitePoint(direction)) return null;
    const a=rail.start,b=rail.end,dx=b[0]-a[0],dy=b[1]-a[1];
    const denominator=dx*direction[1]-dy*direction[0];
    if (Math.abs(denominator)<=1e-10*Math.hypot(dx,dy)*Math.hypot(...direction)) return null;
    const rawT=((point[0]-a[0])*direction[1]-(point[1]-a[1])*direction[0])/denominator;
    // Clamping an arbitrary intersection would change the connector direction.
    // Outside the actual rail segment, there is no capturable guide point.
    if (rawT < -1e-10 || rawT > 1+1e-10) return null;
    const t=Math.max(0,Math.min(1,rawT));
    return {point:[a[0]+t*dx,a[1]+t*dy],t,direction:direction.slice()};
  }

  function rails(register,width,height) {
    const errors=validateQuad(register,width,height);
    if (errors.length) return {valid:false,errors,top:null,bottom:null};
    return {valid:true,errors:[],top:line(register[0],register[3]),bottom:line(register[1],register[2]),side_direction:meanSideDirection(register)};
  }

  // Membership validation for saved points only; the live cursor uses projectAlongDirection.
  function projectToRail(point,rail,scale=[1,1]) {
    if (!finitePoint(point) || !rail || !finitePoint(rail.start) || !finitePoint(rail.end) || !scale.every(x=>Number.isFinite(x)&&x>0)) return null;
    const a=rail.start,b=rail.end,dx=b[0]-a[0],dy=b[1]-a[1];
    // Weighted projection minimizes Euclidean distance in the displayed canvas.
    const denominator=(dx*scale[0])**2+(dy*scale[1])**2;
    if (!(denominator>1e-14)) return null;
    const rawT=((point[0]-a[0])*dx*scale[0]**2+(point[1]-a[1])*dy*scale[1]**2)/denominator;
    const t=Math.max(0,Math.min(1,rawT));
    return {point:[a[0]+t*dx,a[1]+t*dy],t,clamped:t!==rawT};
  }

  function candidate(points,rawPoint,count,width,height) {
    if (!inside(rawPoint,width,height) || ![5,6].includes(count) || points.length>=manualTotal(count)) return null;
    if (points.length<4) return {point:rawPoint.slice(),role:roles[points.length],stage:'register',rail:null};
    const guide=rails(points.slice(0,4),width,height);
    if (!guide.valid || !guide.side_direction) return null;
    const role=roleAt(points.length), rail=(role==='TL'||role==='TR')?'top':'bottom';
    const projected=projectAlongDirection(rawPoint,guide[rail],guide.side_direction.vector);
    return projected?{...projected,role,stage:'digit',rail,digit_index:points.length<6?0:1+Math.floor((points.length-6)/4),
      starts_manual_fallback:points.length===6}:null;
  }

  function generate(points,count,width,height) {
    const empty={valid:false,errors:[],register_valid:false,register_corners:[],rails:null,
      quadrilaterals:[],expected_points:Array.isArray(points)&&points.length>6?manualTotal(count):6,
      mode:Array.isArray(points)&&points.length>6?'manual_fallback':'automatic',spacing:null};
    if (![5,6].includes(count) || !Array.isArray(points) || points.length>manualTotal(count) || !points.every(p=>inside(p,width,height))) {
      return {...empty,errors:['Invalid point count or image coordinates.']};
    }
    if (points.length<4) return {...empty,register_corners:points.map(p=>p.slice())};
    const register=points.slice(0,4).map(p=>p.slice()), guide=rails(register,width,height);
    const result={...empty,register_corners:register,register_valid:guide.valid,
      rails:guide.valid?{top:guide.top,bottom:guide.bottom,side_direction:guide.side_direction}:null};
    if (!guide.valid) return {...result,errors:guide.errors.map(e=>'Register: '+e)};
    if (!guide.side_direction) result.errors.push('Register: side edges need nonzero downward height to define the guide slope.');
    for (let i=4;i<points.length;i++) {
      const role=roleAt(i), rail=guide[(role==='TL'||role==='TR')?'top':'bottom'];
      const projected=projectToRail(points[i],rail);
      if (!projected || Math.hypot(projected.point[0]-points[i][0],projected.point[1]-points[i][1])>1e-5) {
        result.errors.push(`Digit ${i<6?1:2+Math.floor((i-6)/4)}: ${role} is not on its register rail segment.`);
      }
    }
    if (points.length<6 || result.errors.length) return result;
    if (points.length>6) {
      // Point 7 discards all extrapolation. Retain digit 1 and only completed manual cells.
      const first=[register[0].slice(),register[1].slice(),points[4].slice(),points[5].slice()];
      result.quadrilaterals.push(first);
      for (const error of validateQuad(first,width,height)) result.errors.push('Digit 1: '+error);
      for (let i=6;i+3<points.length;i+=4) {
        const q=points.slice(i,i+4).map(point=>point.slice());
        result.quadrilaterals.push(q);
        for (const error of validateQuad(q,width,height)) result.errors.push(`Digit ${2+(i-6)/4}: ${error}`);
      }
      result.valid=points.length===manualTotal(count) && result.errors.length===0;
      return result;
    }
    const interpolate=(rail,t)=>rail.start.map((v,j)=>v+t*(rail.end[j]-v));
    const widths={top:projectToRail(points[5],guide.top).t,bottom:projectToRail(points[4],guide.bottom).t};
    const spacing={};
    for (const name of ['top','bottom']) {
      const rail=guide[name], length=Math.hypot(rail.end[0]-rail.start[0],rail.end[1]-rail.start[1]);
      const w=widths[name], gap=(1-count*w)/(count-1);
      if (!(w>1e-7) || gap < -1e-10) {
        result.errors.push(`Spacing: first digit ${name} width must be positive and fit ${count} equal cells inside the register. Undo (C) to adjust it.`);
      }
      // Clamp floating-point roundoff only. Materially negative gaps are invalid.
      spacing[name]={register_length_px:length,cell_width_fraction:w,
        gap_fraction:Math.max(0,gap),cell_width_px:w*length,gap_px:Math.max(0,gap)*length};
    }
    if (result.errors.length) return result;
    result.spacing=spacing;
    for (let i=0;i<count;i++) {
      const boundary=(name,right)=>{
        const s=spacing[name], t=i*(s.cell_width_fraction+s.gap_fraction)+(right?s.cell_width_fraction:0);
        if (i===0) return right?points[name==='top'?5:4].slice():guide[name].start.slice();
        if (i===count-1 && right) return guide[name].end.slice();
        return interpolate(guide[name],t);
      };
      const q=[boundary('top',false),boundary('bottom',false),boundary('bottom',true),boundary('top',true)];
      result.quadrilaterals.push(q);
      for (const error of validateQuad(q,width,height)) result.errors.push(`Digit ${i+1}: ${error}`);
    }
    result.valid=points.length===6 && result.errors.length===0;
    return result;
  }

  const undo=points=>points.slice(0,-1).map(p=>p.slice());
  return {roles,roleAt,manualTotal,displayToImage,validateQuad,rails,meanSideDirection,projectToRail,projectAlongDirection,candidate,generate,undo};
});
