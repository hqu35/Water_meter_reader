"""Conversion boundary / safety-gate tests. Tiny arrays are unit fixtures, not training data."""
import importlib.util, io, json, tempfile, unittest, zipfile
from pathlib import Path
import numpy as np
from PIL import Image
from types import SimpleNamespace

p=Path(__file__).resolve().parents[1]/'scripts/stage1.py'
spec=importlib.util.spec_from_file_location('stage1',p);s=importlib.util.module_from_spec(spec);spec.loader.exec_module(s)

class ConversionTests(unittest.TestCase):
 def test_one_pixel_and_far_edge(self):
  fg=np.zeros((7,11),dtype=bool);fg[6,10]=True
  self.assertEqual(s.bbox_from_mask(fg),[10,6,11,7])
  cx,cy,w,h=s.normalize_box(s.bbox_from_mask(fg),11,7)
  np.testing.assert_allclose([(cx-w/2)*11,(cy-h/2)*7,(cx+w/2)*11,(cy+h/2)*7],[10,6,11,7])
 def test_disconnected_foreground_is_not_discarded(self):
  fg=np.zeros((20,30),dtype=bool);fg[3:5,4:10]=True;fg[18,28]=True
  self.assertEqual(s.bbox_from_mask(fg),[4,3,29,19])
 def test_empty_is_explicit(self):self.assertIsNone(s.bbox_from_mask(np.zeros((3,4),dtype=bool)))
 def test_full_image_is_valid(self):self.assertEqual(s.normalize_box([0,0,512,512],512,512),[.5,.5,1,1])
 def test_invalid_box_fails(self):
  with self.assertRaises(AssertionError):s.normalize_box([0,0,513,512],512,512)
 def test_pairing_uses_full_parent_not_stem(self):
  with tempfile.TemporaryDirectory() as t:
   p=Path(t)/'d.zip'
   with zipfile.ZipFile(p,'w') as z:
    for split in ['train','test']:
     for d in ['det_img','seg_label']:z.writestr(f'detection/{split}/{d}/same.png',b'fixture')
   with zipfile.ZipFile(p) as z:
    pairs=s.discover(z);self.assertIn('/train/',pairs['train'][0][1]);self.assertIn('/test/',pairs['test'][0][1])
 def test_missing_pair_fails(self):
  with tempfile.TemporaryDirectory() as t:
   p=Path(t)/'d.zip'
   with zipfile.ZipFile(p,'w') as z:z.writestr('detection/train/det_img/a.png',b'fixture')
   with zipfile.ZipFile(p) as z:
    with self.assertRaises(ValueError):s.discover(z)
 def test_actual_dryad_archive_names(self):
  with tempfile.TemporaryDirectory() as t:
   p=Path(t)/'d.zip'
   with zipfile.ZipFile(p,'w') as z:
    for split in ['train','test']:
     for suffix in ['img','seg_label']:z.writestr(f'Word-Wheel_Water_Meter_Dataset/detection/{split}/{split}_{suffix}/{split}0.png',b'fixture')
   with zipfile.ZipFile(p) as z:
    pairs=s.discover(z);self.assertEqual(len(pairs['train']),1);self.assertIn('test_seg_label',pairs['test'][0][1])
 def test_dimension_mismatch_fails(self):
  def png(size):
   b=io.BytesIO();Image.new('L',size).save(b,format='PNG');return b.getvalue()
  with tempfile.TemporaryDirectory() as t:
   p=Path(t)/'d.zip'
   with zipfile.ZipFile(p,'w') as z:z.writestr('i.png',png((4,4)));z.writestr('m.png',png((3,4)))
   with zipfile.ZipFile(p) as z:
    with self.assertRaises(AssertionError):s.decode(z,'i.png','m.png')
 def test_train_gate_cannot_approve_itself(self):
  with tempfile.TemporaryDirectory() as t:
   with self.assertRaises(FileNotFoundError):s.gate(Path(t))
 def test_manifest_mutation_revokes_review(self):
  with tempfile.TemporaryDirectory() as t:
   root=Path(t);art=root/'artifacts';art.mkdir();manifest=art/'split_manifest.json';manifest.write_text('{}')
   s.dump(art/'annotation_review.json',dict(approved=True,reviewed_image_count=32,manifest_sha256=s.sha(manifest),reviewed_sheets=[]))
   manifest.write_text('{"modified":true}')
   with self.assertRaises(AssertionError):s.gate(root)
 def test_prediction_calls_are_bounded_and_keep_manifest_order(self):
  class Model:
   def __init__(self):self.batch_sizes=[]
   def predict(self,source,**kwargs):
    self.batch_sizes.append(len(source));return iter(SimpleNamespace(path=p) for p in source)
  model=Model();rows=[dict(image_file=f'{i}.jpg') for i in [5,2,9,1,7]]
  pairs=list(s.prediction_batches(model,rows,Path('/images'),batch_size=2))
  self.assertEqual(model.batch_sizes,[2,2,1]);self.assertEqual([r['image_file'] for r,p in pairs],[r['image_file'] for r in rows])
 def test_prediction_reordering_is_rejected(self):
  class Model:
   def predict(self,source,**kwargs):return iter(SimpleNamespace(path=p) for p in reversed(source))
  with self.assertRaises(AssertionError):list(s.prediction_batches(Model(),[dict(image_file='a.jpg'),dict(image_file='b.jpg')],Path('/images')))

if __name__=='__main__':unittest.main()
