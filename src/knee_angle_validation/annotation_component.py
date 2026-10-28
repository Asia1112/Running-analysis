"""Streamlit CCv2 component for unbiased three-point reference annotation."""

from collections.abc import Callable

import streamlit as st


_ANNOTATOR = st.components.v2.component(
    "knee_reference_annotator",
    html="""
<div class="toolbar">
  <strong id="status">Biodro (1/3)</strong>
  <div><button id="undo" type="button" title="Cofnij" aria-label="Cofnij punkt">↶</button><button id="reset" type="button" title="Wyczyść" aria-label="Wyczyść punkty">×</button></div>
</div>
<div class="stage"><img id="frame" alt="Klatka do anotacji"><canvas id="overlay"></canvas></div>
""",
    css="""
:host { color: var(--st-text-color); font-family: var(--st-font); }
.toolbar { display:flex; justify-content:space-between; align-items:center; margin-bottom:8px; }
button { width:38px; height:34px; margin-left:6px; border:1px solid var(--st-border-color); border-radius:6px; background:var(--st-secondary-background-color); color:var(--st-text-color); font-size:20px; cursor:pointer; }
.stage { position:relative; width:100%; line-height:0; overflow:hidden; background:#111; }
img { display:block; width:100%; height:auto; }
canvas { position:absolute; inset:0; width:100%; height:100%; cursor:crosshair; }
""",
    js="""
export default function(component) {
  const { data, parentElement, setStateValue } = component
  const image = parentElement.querySelector('#frame')
  const canvas = parentElement.querySelector('#overlay')
  const status = parentElement.querySelector('#status')
  const undo = parentElement.querySelector('#undo')
  const reset = parentElement.querySelector('#reset')
  const labels = Array.isArray(data.point_labels) ? data.point_labels : ['Biodro', 'Kolano', 'Kostka']
  let points = Array.isArray(data.points) ? data.points.slice(0, labels.length).map(p => ({...p})) : []

  function draw() {
    const rect = image.getBoundingClientRect()
    if (!rect.width || !image.naturalWidth) return
    canvas.width = Math.round(rect.width)
    canvas.height = Math.round(rect.width * image.naturalHeight / image.naturalWidth)
    const ctx = canvas.getContext('2d')
    ctx.clearRect(0, 0, canvas.width, canvas.height)
    const sx = canvas.width / image.naturalWidth
    const sy = canvas.height / image.naturalHeight
    if (points.length === 3) {
      ctx.strokeStyle = '#ff7a2f'; ctx.lineWidth = 3; ctx.beginPath()
      ctx.moveTo(points[0].x * sx, points[0].y * sy)
      ctx.lineTo(points[1].x * sx, points[1].y * sy)
      ctx.lineTo(points[2].x * sx, points[2].y * sy); ctx.stroke()
    }
    points.forEach((point, index) => {
      ctx.fillStyle = {'Biodro':'#43a5ff', 'Kolano':'#ff7a2f', 'Kostka':'#65d44d'}[labels[index]]
      ctx.beginPath(); ctx.arc(point.x * sx, point.y * sy, 6, 0, Math.PI * 2); ctx.fill()
    })
    status.textContent = points.length < labels.length ? `${labels[points.length]} (${points.length + 1}/${labels.length})` : 'Anotacja kompletna'
  }
  image.onload = draw
  if (image.src !== data.image) image.src = data.image
  else draw()
  canvas.onclick = (event) => {
    if (points.length >= labels.length || !image.naturalWidth) return
    const rect = canvas.getBoundingClientRect()
    if (!rect.width || !rect.height) return
    points.push({x:(event.clientX-rect.left)*image.naturalWidth/rect.width, y:(event.clientY-rect.top)*image.naturalHeight/rect.height})
    setStateValue('points', points.map(p => ({...p}))); draw()
  }
  undo.onclick = () => { points.pop(); setStateValue('points', points.map(p => ({...p}))); draw() }
  reset.onclick = () => { points = []; setStateValue('points', points); draw() }
  const observer = new ResizeObserver(draw); observer.observe(image)
  return () => observer.disconnect()
}
""",
)


def annotate_image(
    image_data_url: str,
    points: list[dict[str, float]] | None = None,
    *,
    key: str,
    on_points_change: Callable[[], None] | None = None,
    point_labels: list[str] | None = None,
) -> object:
    """Mount the reference annotator and return its component state."""
    if points is None:
        state = st.session_state.get(key, {})
        points = state.get("points", [])
    return _ANNOTATOR(
        data={"image": image_data_url, "points": points or [],
              "point_labels": point_labels if point_labels is not None else ["Biodro", "Kolano", "Kostka"]},
        key=key,
        height="content",
        on_points_change=on_points_change or (lambda: None),
    )
