// Isolated event/state tests. These do not replace a real browser layout test.
import { readFileSync } from 'node:fs';
import assert from 'node:assert/strict';
import test from 'node:test';

const source = readFileSync(new URL('../src/knee_angle_validation/annotation_component.py', import.meta.url), 'utf8');
const javascript = source.split('    js="""')[1].split('""",')[0];
const render = new Function('ResizeObserver', javascript.replace('export default function(component)', 'function render(component)') + '; return render;')(
  class { observe() {} disconnect() {} }
);

function fixture() {
  const rect = { left: 0, top: 0, width: 960, height: 540 };
  const image = { src: '', naturalWidth: 1920, naturalHeight: 1080, getBoundingClientRect: () => rect };
  const ctx = { clearRect() {}, beginPath() {}, moveTo() {}, lineTo() {}, stroke() {}, arc() {}, fill() {} };
  const canvas = { getBoundingClientRect: () => rect, getContext: () => ctx };
  const status = {}, undo = {}, reset = {};
  const elements = { '#frame': image, '#overlay': canvas, '#status': status, '#undo': undo, '#reset': reset };
  const events = [];
  const mount = (points = []) => {
    render({ data: {image: 'data:image/png;base64,test', points},
      parentElement: {querySelector: id => elements[id]},
      setStateValue: (key, value) => events.push({key, value}) });
    image.onload();
  };
  return { mount, events, canvas, image, rect, undo, reset, status };
}

test('three clicks survive Python-style rehydration and use natural-image pixels', () => {
  const f = fixture();
  f.mount();
  f.canvas.onclick({clientX: 10, clientY: 20});
  const first = f.events.at(-1).value;
  assert.deepEqual(first, [{x: 20, y: 40}]);
  f.mount(first);
  f.canvas.onclick({clientX: 30, clientY: 40});
  f.mount(f.events.at(-1).value);
  f.canvas.onclick({clientX: 50, clientY: 60});
  assert.equal(f.events.at(-1).value.length, 3);
  assert.equal(first.length, 1); // Later clicks never mutate earlier payloads.
  assert.equal(f.status.textContent, 'Anotacja kompletna');
  const count = f.events.length;
  f.canvas.onclick({clientX: 100, clientY: 100});
  assert.equal(f.events.length, count);
  f.undo.onclick();
  assert.equal(f.events.at(-1).value.length, 2);
  f.mount(f.events.at(-1).value);
  f.reset.onclick();
  assert.deepEqual(f.events.at(-1).value, []);
  f.mount([]); // Independent session starts blank.
  assert.equal(f.status.textContent, 'Biodro (1/3)');
});

test('resizing the display preserves image coordinates', () => {
  const f = fixture();
  f.rect.width = 480; f.rect.height = 270;
  f.mount();
  f.canvas.onclick({clientX: 10, clientY: 20});
  assert.deepEqual(f.events.at(-1).value, [{x: 40, y: 80}]);
});
