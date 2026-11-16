/* Six-point automatic proposals; point seven starts manual per-cell fallback. */
(() => {
  'use strict';
  const G=window.GuidedRegisterGeometry;
  const data=JSON.parse(document.getElementById('dataset').textContent);
  const items=data.images, byId=new Map(items.map((item,i)=>[item.image_id,i]));
  const prefix='guided-hybrid-cells-v1:'+data.fingerprint+':';
  const $=id=>document.getElementById(id);
  const canvas=$('canvas'), ctx=canvas.getContext('2d');
  const statuses=new Map(), legacySources=new Map();
  let index=0, points=[], image=null, pointer=null, token=0, scheduled=false;
  let storageOK=true, blocked=false, warning='';
  const copy=ps=>ps.map(p=>p.slice());
  const total=item=>G.manualTotal(item.digit_count);
  const legacyPrefix='guided-equal-spacing-v1:'+data.fingerprint+':';

  function makeRecord(item, ps) {
    const geometry=G.generate(ps,item.digit_count,item.width,item.height);
    const status=ps.length===0?'unstarted':geometry.errors.length?'invalid':geometry.valid?'complete':'in_progress';
    return {
      schema:'watermeter-guided-hybrid-cells-v1', annotation_tool:'register_rail_hybrid_cells_browser',
      image_id:item.image_id, source_image:item.source_image, source_image_sha256:item.sha256,
      dataset_group:item.dataset_group, original_image_size_wh:[item.width,item.height],
      corrected_image_size_wh:[item.width,item.height], orientation_label:item.orientation,
      orientation_label_evidence:item.orientation_evidence,
      orientation_correction:item.orientation==='upside_down'?'rotate_180':'identity',
      digit_count:item.digit_count, digit_count_source:'author_six_digit_flag',
      author_six_digit_flag:item.six_digit,
      coordinate_space:'full_resolution_orientation_corrected_image',
      register_point_order:G.roles.slice(), digit_point_order:G.roles.slice(),
      manual_points:copy(ps), register_corners:copy(ps.slice(0,4)),
      manual_point_roles:ps.map((_,i)=>G.roleAt(i)),
      first_digit_right_points:copy(ps.slice(4,6)),
      remaining_digit_manual_points:copy(ps.slice(6)),
      annotation_mode:geometry.mode,
      previous_equal_spacing_record:legacySources.get(item.image_id)||null,
      digit_points:geometry.quadrilaterals.flat().map(p=>p.slice()),
      digit_quadrilaterals:geometry.quadrilaterals,
      spacing:geometry.spacing,
      generated_annotations_unverified:geometry.mode==='automatic',
      digit_box_sources:geometry.quadrilaterals.map((_,i)=>i===0?'manual_with_register_left_anchor':geometry.mode==='automatic'?'equal_spacing_prediction':'manual'),
      guide_rails:geometry.rails,
      geometry:{method:geometry.mode==='automatic'?'equal_width_fitted_gap_per_register_rail':'manual_remaining_cells_on_register_rails',
        projection:'along_mean_register_side_dx_dy_to_rail_segment',
        guide_direction:geometry.rails?geometry.rails.side_direction:null,
        manual_input_order:ps.map((_,i)=>i<4?'register_'+G.roleAt(i):i<6?'digit1_'+G.roleAt(i):'digit'+(2+Math.floor((i-6)/4))+'_'+G.roleAt(i)),
        width_source:geometry.mode==='automatic'?'first_digit_right_corners_and_register_left_edge':'individual_manual_corners_per_cell',
        gap_source:geometry.mode==='automatic'?'(register_rail_length - digit_count * first_cell_width) / (digit_count - 1)':null,
        fallback_trigger:'seventh_saved_manual_point',
        endpoint_anchors:geometry.mode==='automatic'?'first_digit_left_and_last_digit_right_equal_register_sides':'first_digit_left_equals_register_left; all remaining corners manually selected',
        constraints:{TL:'top',BL:'bottom',BR:'bottom',TR:'top'},
        image_warp_applied:false, shared_cell_boundaries_enforced:false, equal_width_per_rail:geometry.mode==='automatic', equal_gap_per_rail:geometry.mode==='automatic'},
      expected_point_count:geometry.expected_points, status,
      validation:{valid:geometry.valid,errors:geometry.errors}
    };
  }

  function validSaved(record,item) {
    if (!record || record.schema!=='watermeter-guided-hybrid-cells-v1' ||
        record.annotation_tool!=='register_rail_hybrid_cells_browser' ||
        record.image_id!==item.image_id || record.source_image_sha256!==item.sha256 ||
        record.orientation_label!==item.orientation || record.digit_count!==item.digit_count ||
        !Array.isArray(record.manual_points)) return false;
    const ps=record.manual_points;
    if (ps.length>total(item) || !ps.every(p=>Array.isArray(p)&&p.length===2&&p.every(Number.isFinite)&&
      p[0]>=0&&p[0]<item.width&&p[1]>=0&&p[1]<item.height)) return false;
    const geometry=G.generate(ps,item.digit_count,item.width,item.height);
    // Invalid cell shapes can be resumed and undone; off-rail data cannot be imported.
    return ps.length<=4 || geometry.register_valid && !geometry.errors.some(e=>e.includes('not on its register rail'));
  }

  function validLegacy(record,item) {
    if (!record || record.schema!=='watermeter-guided-equal-spacing-v1' ||
        record.annotation_tool!=='register_rail_equal_spacing_browser' || !Array.isArray(record.manual_points) ||
        record.manual_points.length>8) return false;
    // Old points 7/8 were diagnostics: validate their rails but never use them as manual corners.
    return validSaved({...record,schema:'watermeter-guided-hybrid-cells-v1',
      annotation_tool:'register_rail_hybrid_cells_browser'},item);
  }

  function getRecord(item) {
    const raw=localStorage.getItem(prefix+'image:'+item.image_id);
    if (raw!==null) {
      const record=JSON.parse(raw);
      if (!validSaved(record,item)) throw Error('Saved state differs from the hybrid schema or image: '+item.image_id);
      if (record.previous_equal_spacing_record) legacySources.set(item.image_id,record.previous_equal_spacing_record);
      return makeRecord(item,record.manual_points);
    }
    const oldRaw=localStorage.getItem(legacyPrefix+'image:'+item.image_id);
    if (oldRaw===null) return null;
    const old=JSON.parse(oldRaw);
    if (!validLegacy(old,item)) throw Error('Previous equal-spacing record is invalid: '+item.image_id);
    legacySources.set(item.image_id,old);
    // Reuse the original six geometry inputs, retaining the complete old record verbatim.
    return makeRecord(item,old.manual_points.slice(0,6));
  }

  function setRecord(item,ps) {
    if (!storageOK) throw Error('Browser storage is unavailable.');
    const old=localStorage.getItem(prefix+'image:'+item.image_id);
    if (old!==null && !validSaved(JSON.parse(old),item)) throw Error('Refusing to overwrite incompatible saved data.');
    const record=makeRecord(item,ps);
    localStorage.setItem(prefix+'image:'+item.image_id,JSON.stringify(record));
    statuses.set(item.image_id,record.status);
    return record;
  }

  function scanProgress() {
    statuses.clear();
    for (const item of items) {
      const record=getRecord(item);
      if (record) statuses.set(item.image_id,record.status);
    }
  }

  function updateState() {
    const item=items[index], record=makeRecord(item,points);
    $('image-meta').textContent=`Image ${index+1} / ${items.length} · ${item.image_id} · ${item.dataset_group}`;
    $('image-details').textContent=`Digits: ${item.digit_count} · Orientation: ${item.orientation==='upside_down'?'ROTATE 180°':'UPRIGHT'} · ${item.width} × ${item.height} original pixels`;
    if (points.length<4) {
      $('step').textContent=`Register outline · ${G.roles[points.length]} · Corner ${points.length+1} / 4`;
    } else if (!G.generate(points,item.digit_count,item.width,item.height).register_valid || record.validation.errors.some(e=>e.startsWith('Register:'))) {
      $('step').textContent='Invalid register outline — Undo (C) or Reset (R) to correct it.';
    } else if (points.length===6) {
      $('step').textContent=record.validation.valid?
        'AUTO — proposals saved. Next (.) keeps them; Space at digit 2 TL discards predictions and starts MANUAL mode.':
        'AUTO spacing invalid — adjust digit 1 with C, or press Space at digit 2 TL to start MANUAL mode.';
    } else if (points.length<total(item)) {
      const role=G.roleAt(points.length), digit=points.length<6?1:2+Math.floor((points.length-6)/4);
      $('step').textContent=`${points.length<6?'Digit 1 right edge':'MANUAL digit '+digit+' / '+item.digit_count} · ${role} · ${(role==='TL'||role==='TR')?'TOP':'BOTTOM'} rail · ${points.length} points saved`;
    } else {
      $('step').textContent=record.validation.valid?'COMPLETE — all remaining digit boxes manually marked':'INVALID — Undo (C) or Reset (R) to correct the points.';
    }
    $('warning').textContent=warning || record.validation.errors.join('\n');
    const fitted=G.generate(points,item.digit_count,item.width,item.height);
    let info=fitted.mode==='manual_fallback'?
      `MANUAL mode: automatic boxes removed. Mark every remaining digit TL → BL → BR → TR. ${fitted.quadrilaterals.length} / ${item.digit_count} complete boxes; ${total(item)} total points required.`:
      fitted.spacing?`AUTO mode: top gap ${fitted.spacing.top.gap_px.toFixed(2)} px; bottom gap ${fitted.spacing.bottom.gap_px.toFixed(2)} px. Moving the mouse does not switch modes; saving point 7 does.`:
      'First digit left corners come from the register outline. Mark its BR and TR to preview automatic boxes.';
    if (record.previous_equal_spacing_record && record.previous_equal_spacing_record.manual_points.length>6) {
      info+=' Previous reference-only points are preserved in the exported original record; they have not been treated as manual boxes.';
    }
    $('spacing-info').textContent=info;
    const counts={complete:0,in_progress:0,invalid:0};
    for (const status of statuses.values()) if (Object.hasOwn(counts,status)) counts[status]++;
    $('progress').textContent=`Complete ${counts.complete} · In progress ${counts.in_progress} · Invalid ${counts.invalid} · Total ${items.length}`;
    $('storage').textContent=storageOK?'Autosaved in this browser. Export JSON for backup.':'Autosave unavailable — export your existing progress.';
    $('storage').className=storageOK?'muted':'storage-error';
  }

  function preview() {
    if (!image || blocked || !pointer) return null;
    const item=items[index], rect=canvas.getBoundingClientRect();
    const raw=G.displayToImage(pointer.x,pointer.y,rect,item.width,item.height);
    if (!raw) return null;
    const guide=G.candidate(points,raw,item.digit_count,item.width,item.height);
    return guide?{...guide,raw}:null;
  }

  function draw() {
    if (!image) return;
    const item=items[index], rect=canvas.getBoundingClientRect();
    if (!(rect.width>0&&rect.height>0)) return;
    const dpr=window.devicePixelRatio||1;
    const backingW=Math.max(1,Math.round(rect.width*dpr)), backingH=Math.max(1,Math.round(rect.height*dpr));
    if (canvas.width!==backingW) canvas.width=backingW;
    if (canvas.height!==backingH) canvas.height=backingH;
    ctx.setTransform(backingW/rect.width,0,0,backingH/rect.height,0,0);
    ctx.clearRect(0,0,rect.width,rect.height);
    ctx.save();
    if (item.orientation==='upside_down') {ctx.translate(rect.width,rect.height);ctx.rotate(Math.PI);}
    ctx.drawImage(image,0,0,rect.width,rect.height);ctx.restore();
    const screen=p=>[p[0]*rect.width/item.width,p[1]*rect.height/item.height];
    function path(ps,color,width=2,closed=false) {
      if (!ps.length) return;
      ctx.beginPath();ps.map(screen).forEach((p,i)=>i?ctx.lineTo(...p):ctx.moveTo(...p));
      if (closed) ctx.closePath();ctx.strokeStyle=color;ctx.lineWidth=width;ctx.stroke();
    }
    const geometry=G.generate(points,item.digit_count,item.width,item.height);
    if (points.length>=4) path(points.slice(0,4),geometry.register_valid?'#f5ce68':'#ff5353',2,true);
    if (geometry.register_valid) {
      path([geometry.rails.top.start,geometry.rails.top.end],'#54dfff',3);
      path([geometry.rails.bottom.start,geometry.rails.bottom.end],'#ff82d3',3);
    }
    geometry.quadrilaterals.forEach((q,i)=>{
      path(q,geometry.errors.some(e=>e.startsWith(`Digit ${i+1}:`))?'#ff5353':'#45e5b1',2,true);
      const [x,y]=screen(q[0]);ctx.fillStyle='#073c30';ctx.font='bold 13px sans-serif';ctx.fillText(String(i+1),x+5,y+16);
    });
    if (points.length===5) path([points[1],points[4]],'#f5ce68',2);
    if (points.length>6) {
      const partial=points.slice(6+Math.floor((points.length-6)/4)*4);
      if (partial.length) path(partial,'#f5ce68',2);
    }
    points.forEach((p,i)=>{
      const [x,y]=screen(p);ctx.beginPath();ctx.arc(x,y,3.5,0,2*Math.PI);
      ctx.fillStyle='#ffd54e';ctx.fill();ctx.strokeStyle='#152235';ctx.lineWidth=1;ctx.stroke();
      ctx.font='bold 11px sans-serif';ctx.fillStyle='#ffe78e';
      const label=i<4?'R-'+G.roles[i]:'D'+(i<6?1:2+Math.floor((i-6)/4))+'-'+G.roleAt(i);
      ctx.fillText(label,x+6,y-6);
    });
    const guide=preview();
    if (guide) {
      const [x,y]=screen(guide.point), [rawX,rawY]=screen(guide.raw);
      const color=guide.rail==='bottom'?'#ff82d3':guide.rail==='top'?'#54dfff':'#83ff9c';
      if (guide.stage==='digit') {
        ctx.setLineDash([4,4]);ctx.beginPath();ctx.moveTo(rawX,rawY);ctx.lineTo(x,y);
        ctx.strokeStyle='#e7efff';ctx.lineWidth=1;ctx.stroke();ctx.setLineDash([]);
        ctx.beginPath();ctx.arc(rawX,rawY,2,0,2*Math.PI);ctx.fillStyle='#c1c9d6';ctx.fill();
      }
      // The crosshair intersection is exactly the point saved by Space.
      ctx.strokeStyle='#071724';ctx.lineWidth=4;ctx.beginPath();
      ctx.moveTo(x-11,y);ctx.lineTo(x+11,y);ctx.moveTo(x,y-11);ctx.lineTo(x,y+11);ctx.stroke();
      ctx.strokeStyle=color;ctx.lineWidth=2;ctx.stroke();
      ctx.beginPath();ctx.arc(x,y,4.5,0,2*Math.PI);ctx.stroke();
    }
  }

  function requestDraw() {
    if (scheduled) return;
    scheduled=true;
    requestAnimationFrame(()=>{scheduled=false;draw();});
  }

  function fitCanvas() {
    if (!image) return;
    const item=items[index], frame=$('frame');
    const scale=Math.min((frame.clientWidth-20)/item.width,(frame.clientHeight-20)/item.height,12);
    canvas.style.width=Math.max(1,item.width*scale)+'px';
    canvas.style.height=Math.max(1,item.height*scale)+'px';draw();
  }

  function show(nextIndex) {
    if (nextIndex<0 || nextIndex>=items.length) return;
    const request=++token;index=nextIndex;points=[];image=null;warning='';blocked=false;
    try {
      const saved=getRecord(items[index]);points=saved?copy(saved.manual_points):[];
      localStorage.setItem(prefix+'position',items[index].image_id);
    } catch(e) {blocked=true;warning=e.message;updateState();return;}
    updateState();
    const next=new Image();
    next.onload=()=>{
      if (request!==token) return;
      const item=items[index];
      if (next.naturalWidth!==item.width || next.naturalHeight!==item.height) {
        blocked=true;warning='Embedded image dimensions do not match the source.';updateState();return;
      }
      image=next;fitCanvas();
    };
    next.onerror=()=>{if(request===token){blocked=true;warning='Embedded image failed to load.';updateState();}};
    next.src=items[index].src;
  }

  function persist(next) {
    if (blocked || !image) return;
    try {setRecord(items[index],next);points=copy(next);warning='';updateState();draw();}
    catch(e) {storageOK=false;blocked=true;warning='Could not save: '+e.message;updateState();}
  }

  function addPoint() {
    if (blocked || !image || points.length>=total(items[index])) return;
    const guide=preview();
    if (!guide) {
      const item=items[index];
      const geometry=G.generate(points,item.digit_count,item.width,item.height);
      const raw=pointer?G.displayToImage(pointer.x,pointer.y,canvas.getBoundingClientRect(),item.width,item.height):null;
      warning=points.length>=4&&(!geometry.register_valid||geometry.errors.some(e=>e.startsWith('Register:')))?
        'Correct the register outline with Undo or Reset before marking digits.':
        raw&&points.length>=4?'Move within the register side limits so the tilted guide intersects the selected rail.':'Move the mouse over the image before pressing Space.';
      updateState();return;
    }
    persist([...points,guide.point]);
  }

  function navigate(delta) {
    if (blocked || !image) return;
    const target=index+delta;
    if (target<0 || target>=items.length) return;
    try {if(points.length)setRecord(items[index],points);show(target);}
    catch(e) {warning=e.message;updateState();}
  }
  function undo() {if(points.length)persist(G.undo(points));}
  function reset() {if(!blocked&&confirm('Remove this image\'s register outline and all digit points?'))persist([]);}
  function movePointer(e) {pointer={x:e.clientX,y:e.clientY};requestDraw();}
  document.addEventListener('pointermove',movePointer);
  document.addEventListener('pointerdown',movePointer);
  window.addEventListener('blur',()=>{pointer=null;requestDraw();});
  window.addEventListener('scroll',requestDraw,{passive:true});
  window.addEventListener('resize',fitCanvas);
  document.addEventListener('keydown',e=>{
    if (e.target.closest('input,select,textarea,[contenteditable=true]')||e.ctrlKey||e.metaKey||e.altKey) return;
    const key=e.key.toLowerCase();
    const fn=e.code==='Space'||e.key===' '?addPoint:
      !e.shiftKey&&(e.code==='Comma'||e.key===',')?()=>navigate(-1):
      !e.shiftKey&&(e.code==='Period'||e.key==='.')?()=>navigate(1):
      e.code==='KeyC'||key==='c'?undo:e.code==='KeyR'||key==='r'?reset:null;
    if (fn) {e.preventDefault();if(!e.repeat)fn();}
  });
  $('previous').onclick=()=>navigate(-1);$('next').onclick=()=>navigate(1);
  $('undo').onclick=undo;$('reset').onclick=reset;
  $('jump-index').max=items.length;
  $('jump').onclick=()=>{
    const n=Number($('jump-index').value);
    if(!Number.isInteger(n)||n<1||n>items.length){alert('Enter an index from 1 to '+items.length);return;}
    navigate(n-1-index);
  };
  $('jump-id-button').onclick=()=>{
    const id=$('jump-id').value.trim();
    if(!byId.has(id)){alert('Image ID not found.');return;}
    navigate(byId.get(id)-index);
  };
  $('export').onclick=()=>{
    try {
      const payload={schema:'watermeter-guided-hybrid-cells-export-v1',dataset_fingerprint:data.fingerprint,
        total_images:items.length,excluded_original_ids:data.excluded_original_ids,
        coordinate_space:'full_resolution_orientation_corrected_image',annotations:items.map(getRecord).filter(Boolean)};
      const url=URL.createObjectURL(new Blob([JSON.stringify(payload,null,2)],{type:'application/json'}));
      const a=document.createElement('a');a.href=url;a.download='watermeter_hybrid_cell_annotations.json';
      document.body.appendChild(a);a.click();a.remove();setTimeout(()=>URL.revokeObjectURL(url),1000);
    } catch(e) {alert('Export failed: '+e.message);}
  };
  // Older storage namespaces stay untouched. Export them as separate legacy datasets.
  const legacyFormats=[
    {prefix:'guided-equal-spacing-v1:',schema:'watermeter-guided-equal-spacing-v1',exportSchema:'watermeter-guided-equal-spacing-export-v1',coordinateSpace:'full_resolution_orientation_corrected_image'},
    {prefix:'guided-register-cells-v1:',schema:'watermeter-guided-cells-v1',exportSchema:'watermeter-guided-cells-export-v1',coordinateSpace:'full_resolution_orientation_corrected_image'},
    {prefix:'manual-all-cells-v1:',schema:'watermeter-manual-cells-v1',exportSchema:'watermeter-manual-cells-export-v1',coordinateSpace:'full_resolution_orientation_corrected_image_pixel_centers'}
  ];
  try {$('legacy-export').hidden=!legacyFormats.some(format=>items.some(item=>localStorage.getItem(format.prefix+data.fingerprint+':image:'+item.image_id)!==null));}
  catch (_) {$('legacy-export').hidden=true;}
  $('legacy-export').onclick=()=>{
    try {
      for (const format of legacyFormats) {
        const annotations=[];
        for (const item of items) {
          const raw=localStorage.getItem(format.prefix+data.fingerprint+':image:'+item.image_id);
          if (raw===null) continue;
          const record=JSON.parse(raw);
          if (record.schema!==format.schema || record.image_id!==item.image_id || record.source_image_sha256!==item.sha256) {
            throw Error('An older record is corrupt or mismatched; it has not been changed.');
          }
          annotations.push(record);
        }
        if (!annotations.length) continue;
        const payload={schema:format.exportSchema,dataset_fingerprint:data.fingerprint,total_images:items.length,
          excluded_original_ids:data.excluded_original_ids,coordinate_space:format.coordinateSpace,annotations};
        const url=URL.createObjectURL(new Blob([JSON.stringify(payload,null,2)],{type:'application/json'}));
        const a=document.createElement('a');a.href=url;a.download=format.exportSchema+'_backup.json';
        document.body.appendChild(a);a.click();a.remove();setTimeout(()=>URL.revokeObjectURL(url),1000);
      }
    } catch(e) {alert('Previous annotation export failed: '+e.message);}
  };
  $('import').onclick=()=>$('import-file').click();
  $('import-file').onchange=async e=>{
    const file=e.target.files[0];if(!file)return;
    try {
      const payload=JSON.parse(await file.text());
      const legacyImport=payload.schema==='watermeter-guided-equal-spacing-export-v1';
      if((!legacyImport&&payload.schema!=='watermeter-guided-hybrid-cells-export-v1')||payload.dataset_fingerprint!==data.fingerprint||
        payload.total_images!==items.length||!Array.isArray(payload.annotations)) throw Error('Not a matching hybrid or previous equal-spacing export. Older annotation formats are preserved separately and cannot be imported.');
      const incoming=[], seen=new Set();
      for(const record of payload.annotations) {
        const item=items[byId.get(record.image_id)];
        if(!item||seen.has(record.image_id)||!(legacyImport?validLegacy(record,item):validSaved(record,item))) throw Error('Invalid, off-rail, duplicate or mismatched record: '+record.image_id);
        seen.add(record.image_id);incoming.push([item,legacyImport?record.manual_points.slice(0,6):record.manual_points,legacyImport?record:record.previous_equal_spacing_record]);
      }
      const conflicts=incoming.filter(([item])=>localStorage.getItem(prefix+'image:'+item.image_id)!==null);
      if(conflicts.length&&!confirm(`Replace ${conflicts.length} saved guided annotation(s)?`))return;
      const backup=incoming.map(([item])=>[prefix+'image:'+item.image_id,localStorage.getItem(prefix+'image:'+item.image_id)]);
      const oldSources=new Map(legacySources);
      try {for(const [item,ps,previous] of incoming){if(previous)legacySources.set(item.image_id,previous);setRecord(item,ps);}}
      catch(err) {
        legacySources.clear();for(const [id,record] of oldSources)legacySources.set(id,record);
        for(const [key,raw] of backup)try{raw===null?localStorage.removeItem(key):localStorage.setItem(key,raw);}catch(_){}
        scanProgress();throw err;
      }
      scanProgress();show(index);
    } catch(err) {alert('Import failed: '+err.message);}
    finally {e.target.value='';}
  };
  try {
    const key=prefix+'storage-check';localStorage.setItem(key,'1');localStorage.removeItem(key);
    scanProgress();const position=localStorage.getItem(prefix+'position')||localStorage.getItem(legacyPrefix+'position');
    if(position&&byId.has(position))index=byId.get(position);
    show(index);
  } catch(e) {storageOK=false;blocked=true;warning='Could not safely load browser storage: '+e.message;updateState();}
  document.body.focus();
})();
