# -*- coding: utf-8 -*-
"""Local browser app for basketball-focused horizontal-to-vertical conversion."""

import cgi
import json
import logging
import math
import os
import re
import shutil
import subprocess
import sys
import tempfile
import threading
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlparse

import cv2

from basketball_cropper import (
    detect_shot_boundaries,
    read_frame,
    render_vertical,
    suggest_player,
    track_target_path,
)


HOST = "127.0.0.1"
PORT = int(os.environ.get("BASKETBALL_CROPPER_PORT", "8765"))
MAX_UPLOAD_BYTES = 1024 * 1024 * 1024
DATA_DIR = os.path.join(tempfile.gettempdir(), "basketball_vertical_studio")
os.makedirs(DATA_DIR, exist_ok=True)

JOBS = {}
TASKS = {}
STATE_LOCK = threading.Lock()
LOG = logging.getLogger("basketball_vertical_studio")
YOLO_WORKER = os.path.join(os.path.dirname(os.path.abspath(__file__)), "basketball_yolo_worker.py")


def _yolo_interpreter():
    configured = os.environ.get("BASKETBALL_YOLO_PYTHON", "").strip()
    if configured:
        return configured if os.path.isfile(configured) else ""
    if sys.version_info >= (3, 8):
        return sys.executable
    return ""


def _check_yolo_backend():
    interpreter = _yolo_interpreter()
    if not interpreter:
        return {
            "available": False,
            "message": (
                "需要单独配置 Python 3.8+ 环境并安装 requirements-yolo.txt；"
                "可通过 BASKETBALL_YOLO_PYTHON 指定该环境的 python.exe。"
            ),
        }
    try:
        result = subprocess.run(
            [interpreter, YOLO_WORKER, "--check"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=20,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired) as error:
        return {"available": False, "message": "YOLO 环境检查失败：{}".format(error)}
    if result.returncode == 0:
        try:
            status = json.loads(result.stdout.decode("utf-8").strip().splitlines()[-1])
        except (IndexError, UnicodeDecodeError, ValueError):
            return {
                "available": False,
                "message": "YOLO worker 返回了无法识别的环境状态。",
            }
        if status.get("event") != "ready":
            return {
                "available": False,
                "message": "YOLO worker 未能确认运行环境。",
            }
        if status.get("cuda_available"):
            message = "YOLOv8 + ByteTrack 已就绪，GPU：{}。".format(
                status.get("device", "CUDA")
            )
        else:
            message = (
                "YOLOv8 + ByteTrack 已就绪，但当前 PyTorch 是 CPU 版；"
                "将使用 CPU。"
            )
        message += (
            "模型权重已就绪。"
            if status.get("model_ready")
            else "首次检测时将下载 YOLO 模型权重。"
        )
        return {
            "available": True,
            "message": message,
            "cuda_available": bool(status.get("cuda_available")),
            "device": status.get("device"),
            "torch_version": status.get("torch_version"),
            "cuda_version": status.get("cuda_version"),
        }
    detail = result.stdout.decode("utf-8", errors="replace").strip()
    if not detail:
        detail = result.stderr.decode("utf-8", errors="replace").strip()
    return {
        "available": False,
        "message": "YOLO 环境尚未就绪。{}".format(detail[-500:]),
    }


def _run_yolo_worker(payload, progress=None):
    interpreter = _yolo_interpreter()
    if not interpreter:
        raise ValueError(
            "YOLOv8 跟踪不可用。请配置 Python 3.8+ 环境，安装 requirements-yolo.txt，"
            "并设置 BASKETBALL_YOLO_PYTHON。"
        )
    with tempfile.TemporaryFile() as error_log:
        process = subprocess.Popen(
            [interpreter, YOLO_WORKER],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=error_log,
        )
        process.stdin.write(json.dumps(payload).encode("utf-8"))
        process.stdin.close()
        result = None
        worker_error = ""
        for line in process.stdout:
            try:
                event = json.loads(line.decode("utf-8"))
            except (UnicodeDecodeError, ValueError):
                continue
            if event.get("event") == "progress" and progress is not None:
                progress(event.get("progress", 0.0))
            elif event.get("event") == "result":
                result = event.get("result")
            elif event.get("event") == "error":
                worker_error = event.get("error", "")
        return_code = process.wait()
        process.stdout.close()
        error_log.seek(0)
        error_detail = error_log.read().decode("utf-8", errors="replace")[-1500:]
    if return_code != 0 or result is None:
        detail = worker_error or error_detail or "YOLO worker 未返回有效结果。"
        raise RuntimeError(detail)
    return result


PAGE = r"""<!doctype html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>篮影 · 篮球横转竖工作台</title>
<style>
:root{--ink:#14211d;--muted:#66756f;--green:#146c4a;--lime:#d9f36a;--paper:#f4f6f1;--line:#e2e8e0;--white:#fff;--orange:#f08a35}
*{box-sizing:border-box}body{margin:0;background:var(--paper);color:var(--ink);font-family:Inter,"Microsoft YaHei",system-ui,sans-serif}
header{height:72px;border-bottom:1px solid var(--line);background:#fff;display:flex;align-items:center;justify-content:space-between;padding:0 max(24px,calc((100vw - 1180px)/2))}
.brand{display:flex;gap:12px;align-items:center;font-weight:800;letter-spacing:.03em}.mark{width:34px;height:34px;background:var(--green);border-radius:11px;display:grid;place-items:center;color:#fff;font-size:19px}
.local{font-size:12px;color:var(--green);background:#eaf4ec;border-radius:99px;padding:8px 12px}
main{max-width:1180px;margin:0 auto;padding:38px 24px 60px}.eyebrow{font-size:12px;letter-spacing:.14em;color:var(--green);font-weight:800;text-transform:uppercase}
h1{font-size:clamp(30px,4vw,46px);letter-spacing:-.04em;margin:10px 0 8px}.subtitle{color:var(--muted);margin:0 0 27px;line-height:1.7}
.layout{display:grid;grid-template-columns:minmax(0,1.6fr) minmax(300px,.85fr);gap:20px}.card{background:var(--white);border:1px solid var(--line);border-radius:18px;padding:20px;box-shadow:0 10px 28px rgba(28,55,41,.035)}
.card h2{font-size:16px;margin:0 0 14px}.drop{border:1px dashed #afc2b3;border-radius:14px;background:#f9fbf8;padding:22px;text-align:center;cursor:pointer;color:var(--muted);transition:.2s}.drop:hover{border-color:var(--green);background:#f0f7ef}.drop strong{display:block;color:var(--ink);margin-bottom:5px}.drop input{display:none}
.preview{position:relative;width:100%;background:#111;border-radius:12px;overflow:hidden;min-height:200px;display:grid;place-items:center;margin-top:16px}.preview img{display:block;max-width:100%;max-height:54vh;object-fit:contain}.preview canvas{position:absolute;left:0;top:0;width:100%;height:100%;touch-action:none;cursor:crosshair}
.placeholder{color:#bac3bd;padding:50px 14px;text-align:center}.timeline{display:flex;align-items:center;gap:12px;margin-top:14px;color:var(--muted);font-size:12px}.timeline input{flex:1;accent-color:var(--green)}.time{min-width:44px;text-align:right;font-variant-numeric:tabular-nums}
.hint{font-size:12px;color:var(--muted);line-height:1.65;margin-top:11px}.badge{display:inline-flex;gap:6px;align-items:center;padding:6px 9px;border-radius:99px;background:#eff4ed;color:var(--green);font-size:11px;font-weight:700}.section{margin-top:20px}.modes{display:grid;grid-template-columns:1fr 1fr;gap:10px}.mode{border:1px solid var(--line);border-radius:12px;padding:12px;cursor:pointer}.mode input{accent-color:var(--green)}.mode strong{display:block;font-size:13px;margin:4px 0}.mode small{color:var(--muted);line-height:1.5;display:block}
.btn{border:0;border-radius:11px;background:var(--green);color:#fff;font-weight:750;padding:12px 16px;cursor:pointer;font-size:14px;width:100%;margin-top:14px;transition:.15s}.btn:hover{background:#0d573a;transform:translateY(-1px)}.btn:disabled{opacity:.45;cursor:not-allowed;transform:none}.btn.secondary{background:#edf4eb;color:var(--green);border:1px solid #dce8dc}.btn.secondary:hover{background:#e0eddf}
.status{font-size:13px;line-height:1.6;color:var(--muted);margin-top:13px;min-height:24px}.status.error{color:#b4382e}.status.warning{color:#78531b}.status.success{color:var(--green)}.progress{height:6px;border-radius:9px;background:#edf1ec;overflow:hidden;margin-top:10px}.progress i{display:block;height:100%;width:0;background:var(--lime);transition:width .25s}
.meta{display:flex;gap:8px;flex-wrap:wrap;margin-top:12px}.meta span{padding:7px 9px;background:#f4f6f2;border-radius:8px;color:var(--muted);font-size:11px}
.warning{background:#fff8e9;border:1px solid #f5e4bd;color:#78531b;padding:11px;border-radius:10px;font-size:11px;line-height:1.55;margin-top:12px}
.shot-picker{display:flex;align-items:center;gap:10px;margin-top:13px;color:var(--muted);font-size:12px}.shot-picker select{flex:1;min-width:0;border:1px solid var(--line);border-radius:9px;background:#fff;color:var(--ink);padding:9px;font:inherit}.shot-picker select:disabled{opacity:.55}
.candidates{display:grid;gap:7px;margin-top:10px}.candidate{display:flex;align-items:center;justify-content:space-between;gap:10px;border:1px solid var(--line);border-radius:9px;background:#fff;color:var(--ink);padding:9px 11px;text-align:left;cursor:pointer;font:inherit;font-size:12px}.candidate:hover{border-color:var(--green)}.candidate.selected{border-color:var(--green);background:#edf6ed;color:var(--green);font-weight:750}.candidate small{color:var(--muted);font-weight:400}.mode.yolo{grid-column:1/-1}.backend-note{font-size:11px;color:var(--muted);line-height:1.5;margin-top:7px}
.tracking-legend{font-size:11px;color:var(--muted);margin-top:8px}.tracking-legend strong{color:var(--green)}
.result{margin-top:14px}.result a{display:block;text-decoration:none;text-align:center;background:var(--lime);color:var(--ink);padding:12px;border-radius:11px;font-weight:800}
.foot{font-size:11px;color:#87928c;text-align:center;margin-top:24px}
@media(max-width:800px){.layout{grid-template-columns:1fr}.preview img{max-height:42vh}main{padding-top:26px}.local{font-size:10px}}
</style></head>
<body>
<header><div class="brand"><span class="mark">篮</span><span>篮影 <span style="font-weight:500;color:#718079">COURTSIDE STUDIO</span></span></div><span class="local">● 仅在本机处理</span></header>
<main><div class="eyebrow">BASKETBALL · SMART REFRAME</div><h1>把比赛焦点，带进竖屏。</h1>
<p class="subtitle">上传横屏篮球视频，锁定并预览目标跟踪，再生成平滑的 9:16 竖屏片段。视频只在本机处理。</p>
<div class="layout">
<section class="card"><h2>01　上传比赛视频</h2>
<label class="drop" id="drop"><strong>选择或拖入视频文件</strong><span>MP4、MOV、AVI 等常见格式，最大 1 GB</span><input id="file" type="file" accept="video/*"></label>
<div id="metadata" class="meta"></div>
<div class="preview" id="preview"><div class="placeholder" id="placeholder">上传视频后，在清晰看到球员的时间点获取候选推荐，或直接拖框选主体</div><img id="frame" hidden><canvas id="overlay" hidden></canvas></div>
<label class="shot-picker">镜头片段<select id="shot" disabled><option>请先上传视频</option></select></label>
<div class="timeline"><span>选取时间</span><input id="seek" type="range" min="0" max="0" step=".05" value="0" disabled><span class="time" id="clock">00:00</span></div>
<div class="hint">先选择一个镜头片段，再在片段内挑选清晰画面确认球员。导出仅包含该镜头片段，避免跟踪框跨转播镜头错位；需要多段时可分别导出。</div>
</section>
<aside class="card"><h2>02　选择主体方式</h2>
<div class="modes"><label class="mode"><input type="radio" name="mode" value="manual"><strong>手动框选</strong><small>直接指定球员，系统从选定时间开始跟踪。</small></label><label class="mode"><input type="radio" name="mode" value="auto" checked><strong>自动推荐 + 手动确认</strong><small>先看候选排序，再点击候选或拖框确认主体。</small></label><label class="mode yolo"><input id="yoloMode" type="radio" name="mode" value="yolo" disabled><strong>YOLOv8 检测 + ByteTrack 跟踪（可选）</strong><small>检测球员后分配跟踪 ID；选定起始球员，后续按其 ID 跟踪。</small></label></div>
<div id="yoloHint" class="backend-note">正在检查可选 YOLOv8 环境…</div>
<button id="detect" class="btn secondary" disabled>分析当前画面并推荐球员</button>
<div id="suggestions" class="candidates" hidden></div>
<button id="track" class="btn secondary" disabled>锁定目标并生成跟踪预览</button>
<div id="trackingLegend" class="tracking-legend"><strong>绿色框</strong>为当前跟踪位置；橙色框表示跟踪位置不可靠或特征不足。检测置信度不等于球员身份置信度。</div>
<div class="warning">镜头切分基于画面颜色变化，是辅助定位而非专业转播切镜识别；候选排序也不是篮球专用模型，不代表持球概率。请确认镜头段和球员，必要时换段重新选择。</div>
<div class="section"><span class="badge">输出　9:16 · 720 × 1280 · MP4</span><button id="export" class="btn" disabled>生成竖屏视频</button><div id="status" class="status">请先上传视频，再确认推荐球员或手动框选主体。</div><div class="progress"><i id="progress"></i></div><div id="result" class="result"></div></div>
</aside></div>
<div class="foot">篮影本地工作台 · 支持光流与 YOLOv8 + ByteTrack 目标跟踪；原视频音轨会保留。</div>
</main>
<script>
const $=id=>document.getElementById(id), file=$('file'), frame=$('frame'), canvas=$('overlay'), ctx=canvas.getContext('2d');
let jobId=null, info=null, box=null, drawing=false, origin=null, suggestions=[], selectedSuggestion=-1, shots=[], tracking=null, trackingTaskId=null, trackingStart=null, trackingBusy=false;
function message(text,kind=''){const el=$('status');el.textContent=text;el.className='status '+kind}
function fmt(t){t=Math.floor(t||0);return String(Math.floor(t/60)).padStart(2,'0')+':'+String(t%60).padStart(2,'0')}
function activeShot(){return shots[Number($('shot').value)]||null}
function isYoloMode(){return document.querySelector('input[name=mode]:checked').value==='yolo'}
function syncCanvas(){if(!frame.naturalWidth)return;const r=frame.getBoundingClientRect(),p=$('preview').getBoundingClientRect();canvas.width=frame.naturalWidth;canvas.height=frame.naturalHeight;canvas.style.left=(r.left-p.left)+'px';canvas.style.top=(r.top-p.top)+'px';canvas.style.width=r.width+'px';canvas.style.height=r.height+'px';draw()}
function trackedBoxAt(t){if(!tracking||t<trackingStart)return null;const index=Math.min(Math.round((t-trackingStart)*info.fps),tracking.boxes.length-1);return index<0?null:{box:tracking.boxes[index],confidence:tracking.confidences[index],found:tracking.found?tracking.found[index]:true}}
function draw(){ctx.clearRect(0,0,canvas.width,canvas.height);suggestions.forEach((player,index)=>{const [x,y,w,h]=player.box;ctx.lineWidth=Math.max(2,canvas.width/700);ctx.strokeStyle=index===selectedSuggestion?'#d9f36a':'#f08a35';ctx.strokeRect(x,y,w,h);ctx.fillStyle=ctx.strokeStyle;ctx.font=`bold ${Math.max(14,canvas.width/55)}px sans-serif`;ctx.fillText(String(index+1),x+4,Math.max(18,y-5))});const tracked=trackedBoxAt(Number($('seek').value));const visibleBox=tracked?tracked.box:tracking?null:box;if(visibleBox){const uncertain=tracked&&(tracked.found===false||tracked.confidence<0.25);ctx.lineWidth=Math.max(3,canvas.width/400);ctx.strokeStyle=uncertain?'#f08a35':'#d9f36a';ctx.fillStyle=uncertain?'rgba(240,138,53,.12)':'rgba(217,243,106,.14)';ctx.fillRect(...visibleBox);ctx.strokeRect(...visibleBox)}}
function renderSuggestions(){const list=$('suggestions');list.replaceChildren();list.hidden=!suggestions.length;suggestions.forEach((player,index)=>{const button=document.createElement('button');button.type='button';button.className='candidate'+(index===selectedSuggestion?' selected':'');button.setAttribute('aria-pressed',String(index===selectedSuggestion));const title=document.createElement('span');title.textContent=(isYoloMode()?'YOLO 球员候选 ':'推荐候选 ')+(index+1);const hint=document.createElement('small');hint.textContent=index===selectedSuggestion?'已确认':(typeof player.confidence==='number'?'置信度 '+player.confidence.toFixed(2):'点击选择');button.append(title,hint);button.addEventListener('click',()=>chooseSuggestion(index));list.appendChild(button)})}
function clearTracking(){tracking=null;trackingTaskId=null;trackingStart=null;refreshExport()}
function chooseSuggestion(index){if(!suggestions[index])return;clearTracking();selectedSuggestion=index;box=suggestions[index].box.slice();renderSuggestions();draw();refreshExport();message(`已选择候选 ${index+1}。请在画面中确认框选范围，也可拖框修正。`,'success')}
function coords(e){const r=canvas.getBoundingClientRect();return [(e.clientX-r.left)*canvas.width/r.width,(e.clientY-r.top)*canvas.height/r.height]}
function loadFrame(){if(!jobId)return;const t=Number($('seek').value);$('clock').textContent=fmt(t);frame.onload=()=>{frame.hidden=false;$('placeholder').hidden=true;canvas.hidden=false;syncCanvas()};frame.src='/api/frame?job='+encodeURIComponent(jobId)+'&t='+t+'&v='+Date.now()}
canvas.addEventListener('pointerdown',e=>{if(!jobId)return;clearTracking();canvas.setPointerCapture(e.pointerId);origin=coords(e);box=[origin[0],origin[1],0,0];selectedSuggestion=-1;drawing=true;renderSuggestions();draw()});
canvas.addEventListener('pointermove',e=>{if(!drawing)return;const p=coords(e);box=[Math.min(origin[0],p[0]),Math.min(origin[1],p[1]),Math.abs(p[0]-origin[0]),Math.abs(p[1]-origin[1])];draw()});
canvas.addEventListener('pointerup',e=>{if(!drawing)return;drawing=false;const p=coords(e);box=[Math.min(origin[0],p[0]),Math.min(origin[1],p[1]),Math.abs(p[0]-origin[0]),Math.abs(p[1]-origin[1])];if(box[2]<8||box[3]<8)box=null;draw();refreshExport()});
window.addEventListener('resize',syncCanvas);$('seek').addEventListener('input',()=>{draw();refreshExport();loadFrame()});
$('shot').addEventListener('change',()=>{const shot=activeShot();if(!shot)return;clearTracking();box=null;suggestions=[];selectedSuggestion=-1;renderSuggestions();draw();$('seek').min=shot.start;$('seek').max=Math.max(shot.start,shot.end-1/info.fps);$('seek').value=shot.start;refreshExport();loadFrame();message(`已选择镜头片段 ${Number($('shot').value)+1}。请确认本段中的目标球员。`)});
function refreshExport(){$('track').disabled=!jobId||!activeShot()||!box||trackingBusy;$('export').disabled=!jobId||!activeShot()||!tracking||!trackingTaskId||trackingBusy;$('detect').disabled=!jobId||document.querySelector('input[name=mode]:checked').value==='manual'||trackingBusy;$('track').textContent=isYoloMode()?'使用 YOLOv8 + ByteTrack 跟踪所选球员':'锁定目标并生成跟踪预览'}
function changeMode(){clearTracking();box=null;suggestions=[];selectedSuggestion=-1;renderSuggestions();draw();refreshExport();message(isYoloMode()?'YOLOv8 将检测球员；请选中 YOLO 检测框，再启动 ByteTrack 跟踪。':'请确认球员主体后生成跟踪预览。')}
document.querySelectorAll('input[name=mode]').forEach(el=>el.addEventListener('change',changeMode));
fetch('/api/capabilities').then(r=>r.json()).then(result=>{const yolo=result.yolo;$('yoloMode').disabled=!yolo.available;$('yoloHint').textContent=yolo.message}).catch(error=>{$('yoloMode').disabled=true;$('yoloHint').textContent='无法检查 YOLO 环境：'+error.message});
$('drop').addEventListener('dragover',e=>{e.preventDefault();$('drop').style.borderColor='#146c4a'});
$('drop').addEventListener('drop',e=>{e.preventDefault();if(e.dataTransfer.files.length)upload(e.dataTransfer.files[0])});
file.addEventListener('change',()=>{if(file.files.length)upload(file.files[0])});
async function upload(f){jobId=null;shots=[];tracking=null;trackingTaskId=null;trackingStart=null;trackingBusy=false;$('shot').disabled=true;$('shot').replaceChildren(new Option('正在分析镜头切换…',''));message('正在读取视频并分析镜头切换…');$('metadata').innerHTML='';box=null;suggestions=[];selectedSuggestion=-1;renderSuggestions();$('result').innerHTML='';$('progress').style.width='0';$('export').disabled=true;$('track').disabled=true;$('detect').disabled=true;
const data=new FormData();data.append('video',f);try{const r=await fetch('/api/upload',{method:'POST',body:data});const j=await r.json();if(!r.ok)throw Error(j.error||'上传失败');jobId=j.job_id;info=j.info;shots=j.shots;$('shot').replaceChildren(...shots.map((shot,index)=>new Option(`镜头片段 ${index+1}　${fmt(shot.start)}–${fmt(shot.end)}`,String(index))));$('shot').disabled=false;$('seek').disabled=false;$('seek').step=(1/info.fps).toFixed(4);$('seek').min=shots[0].start;$('seek').max=Math.max(shots[0].start,shots[0].end-1/info.fps);$('seek').value=shots[0].start;$('metadata').innerHTML=`<span>${f.name.replace(/[<>]/g,'')}</span><span>${info.width} × ${info.height}</span><span>${fmt(info.duration)}</span><span>${info.fps.toFixed(1)} FPS</span><span>${shots.length} 个估算片段</span>`;refreshExport();loadFrame();message(`视频已分析为 ${shots.length} 个镜头片段。请选择片段并确认球员。`,'success')}catch(e){shots=[];$('shot').replaceChildren(new Option('分析失败',''));message(e.message,'error')}}
async function requestSuggestion(){if(!jobId)return;clearTracking();message(isYoloMode()?'正在使用 YOLOv8 检测当前画面中的球员…':'正在分析当前画面中的球员…');$('detect').disabled=true;try{const r=await fetch('/api/suggest',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({job_id:jobId,time:Number($('seek').value),detector:isYoloMode()?'yolov8':'hog'})});const j=await r.json();if(!r.ok)throw Error(j.error||'没有找到球员');suggestions=j.players;selectedSuggestion=0;box=j.players[0].box.slice();renderSuggestions();draw();refreshExport();const cue=isYoloMode()?'候选由 YOLOv8 person 类检测，请选择要跟踪的球员。':(j.ball_cue?'检测到疑似橙色球体，候选按距离排序。':'未检测到明确篮球，候选按画面位置与框大小排序。');message(`找到 ${j.players.length} 个候选球员。${cue}`,'success')}catch(e){box=null;suggestions=[];selectedSuggestion=-1;renderSuggestions();draw();refreshExport();message(e.message,'error')}finally{refreshExport()}}
$('detect').addEventListener('click',requestSuggestion);
async function startTracking(){if(!jobId||!box||!activeShot())return;clearTracking();trackingBusy=true;refreshExport();$('seek').disabled=true;$('shot').disabled=true;canvas.style.pointerEvents='none';$('progress').style.width='0';suggestions=[];selectedSuggestion=-1;renderSuggestions();const start=Number($('seek').value),targetBox=box.map(Math.round),shotIndex=Number($('shot').value),tracker=isYoloMode()?'bytetrack':'optical_flow';message(tracker==='bytetrack'?'正在使用 YOLOv8 + ByteTrack 逐帧关联球员 ID…':'正在逐帧跟踪所选目标，请稍候…');try{const r=await fetch('/api/track',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({job_id:jobId,time:start,shot_index:shotIndex,box:targetBox,tracker})});const result=await r.json();if(!r.ok)throw Error(result.error||'无法启动目标跟踪');const task=await watchTask(result.task_id,'tracking');tracking=task.tracking;trackingTaskId=result.task_id;trackingStart=start;box=targetBox;draw();refreshExport();if(tracking.tracking_warning){message(`${tracking.tracking_warning}可重新选取清晰帧并锁定目标。`,'warning')}else{const detail=tracking.tracker_id===undefined?'':'（球员 ID '+tracking.tracker_id+'）';message(`目标跟踪预览已生成：${tracking.boxes.length} 帧${detail}。拖动时间轴检查跟踪框。`,'success')}}catch(e){clearTracking();message(e.message,'error')}finally{trackingBusy=false;$('seek').disabled=false;$('shot').disabled=false;refreshExport()}}
$('track').addEventListener('click',startTracking);
$('export').addEventListener('click',async()=>{if(!jobId||!trackingTaskId){message('请先锁定目标并检查跟踪预览。','error');return}const payload={job_id:jobId,time:trackingStart,shot_index:Number($('shot').value),box:box.map(Math.round),tracking_task_id:trackingTaskId};$('export').disabled=true;$('detect').disabled=true;$('track').disabled=true;$('result').innerHTML='';$('progress').style.width='0';message('正在按跟踪轨迹生成竖屏片段…');try{const r=await fetch('/api/export',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(payload)});const j=await r.json();if(!r.ok)throw Error(j.error||'无法开始导出');await watchTask(j.task_id,'export')}catch(e){message(e.message,'error');refreshExport()}});
async function watchTask(id,kind){try{while(true){await new Promise(r=>setTimeout(r,700));const r=await fetch('/api/task?id='+encodeURIComponent(id));const j=await r.json();if(!r.ok)throw Error(j.error||'无法获取处理进度');$('progress').style.width=Math.round(j.progress*100)+'%';if(j.status==='failed')throw Error(j.error);if(j.status==='done'){if(kind==='tracking')return j;const warning=j.info.tracking_warning;$('result').innerHTML=`<a href="/api/download?id=${encodeURIComponent(id)}">下载竖屏成片　↓</a>`;message(`已完成：${j.info.width} × ${j.info.height}，${fmt(j.info.duration)}。${warning?warning+'建议回到跟踪预览检查橙色目标框。':''}`,warning?'warning':'success');return j}message(`${kind==='tracking'?'正在计算目标轨迹':'正在导出竖屏视频'}… ${Math.round(j.progress*100)}%`)} }catch(e){message(e.message,'error');throw e}finally{if(kind==='export'){refreshExport();$('detect').disabled=false}}}
</script></body></html>"""


def _job_path(job_id):
    if not re.match(r"^[0-9a-f]{32}$", job_id or ""):
        raise ValueError("无效的视频任务编号。")
    with STATE_LOCK:
        job = JOBS.get(job_id)
    if job is None or not os.path.isfile(job["path"]):
        raise ValueError("视频任务已失效，请重新上传视频。")
    return job


def _json_bytes(value):
    return json.dumps(value, ensure_ascii=False).encode("utf-8")


class RequestHandler(BaseHTTPRequestHandler):
    server_version = "BasketballVerticalStudio/1.0"

    def log_message(self, fmt, *args):
        LOG.info("%s - %s", self.address_string(), fmt % args)

    def _send_json(self, value, status=200):
        payload = _json_bytes(value)
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(payload)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(payload)

    def _read_json(self):
        length = int(self.headers.get("Content-Length", "0"))
        if length <= 0 or length > 1024 * 1024:
            raise ValueError("请求数据为空或过大。")
        try:
            return json.loads(self.rfile.read(length).decode("utf-8"))
        except (UnicodeDecodeError, ValueError):
            raise ValueError("请求数据不是有效 JSON。")

    def do_GET(self):
        parsed = urlparse(self.path)
        query = parse_qs(parsed.query)
        try:
            if parsed.path == "/":
                payload = PAGE.encode("utf-8")
                self.send_response(200)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.send_header("Content-Length", str(len(payload)))
                self.end_headers()
                self.wfile.write(payload)
            elif parsed.path == "/api/capabilities":
                self._send_json({"yolo": _check_yolo_backend()})
            elif parsed.path == "/api/frame":
                job = _job_path(query.get("job", [""])[0])
                seconds = float(query.get("t", ["0"])[0])
                frame, _ = read_frame(job["path"], seconds)
                ok, encoded = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 88])
                if not ok:
                    raise RuntimeError("无法编码预览画面。")
                payload = encoded.tobytes()
                self.send_response(200)
                self.send_header("Content-Type", "image/jpeg")
                self.send_header("Content-Length", str(len(payload)))
                self.send_header("Cache-Control", "no-store")
                self.end_headers()
                self.wfile.write(payload)
            elif parsed.path == "/api/task":
                task_id = query.get("id", [""])[0]
                if not isinstance(task_id, str) or not re.match(r"^[0-9a-f]{32}$", task_id):
                    raise ValueError("无效的导出任务编号。")
                with STATE_LOCK:
                    task = TASKS.get(task_id)
                    if task is None:
                        raise ValueError("导出任务不存在。")
                    response = {
                        "status": task["status"],
                        "progress": task["progress"],
                        "error": task["error"],
                        "info": task["info"],
                        "tracking": (
                            task.get("tracking")
                            if task["status"] == "done"
                            and task.get("kind") == "tracking"
                            else None
                        ),
                    }
                self._send_json(response)
            elif parsed.path == "/api/download":
                task_id = query.get("id", [""])[0]
                with STATE_LOCK:
                    task = TASKS.get(task_id)
                if not task or task["status"] != "done" or not os.path.isfile(task["output"]):
                    raise ValueError("成片尚未准备好或任务已失效。")
                size = os.path.getsize(task["output"])
                self.send_response(200)
                self.send_header("Content-Type", "video/mp4")
                self.send_header("Content-Length", str(size))
                self.send_header(
                    "Content-Disposition",
                    'attachment; filename="basketball_vertical.mp4"',
                )
                self.end_headers()
                with open(task["output"], "rb") as output_file:
                    shutil.copyfileobj(output_file, self.wfile)
            else:
                self._send_json({"error": "页面不存在。"}, 404)
        except (ValueError, OSError) as error:
            self._send_json({"error": str(error)}, 400)
        except (BrokenPipeError, ConnectionResetError):
            LOG.info("Client closed a connection while reading a response.")
        except Exception as error:
            LOG.exception("GET request failed.")
            self._send_json({"error": str(error)}, 500)

    def do_POST(self):
        parsed = urlparse(self.path)
        try:
            if parsed.path == "/api/upload":
                self._upload()
            elif parsed.path == "/api/suggest":
                self._suggest()
            elif parsed.path == "/api/track":
                self._track()
            elif parsed.path == "/api/export":
                self._export()
            else:
                self._send_json({"error": "接口不存在。"}, 404)
        except ValueError as error:
            self._send_json({"error": str(error)}, 400)
        except (BrokenPipeError, ConnectionResetError):
            LOG.info("Client closed a connection while uploading or submitting a task.")
        except Exception as error:
            LOG.exception("POST request failed.")
            self._send_json({"error": str(error)}, 500)

    def _upload(self):
        length = int(self.headers.get("Content-Length", "0"))
        if length <= 0 or length > MAX_UPLOAD_BYTES:
            raise ValueError("视频文件为空或超过 1 GB 限制。")
        form = cgi.FieldStorage(
            fp=self.rfile,
            headers=self.headers,
            environ={
                "REQUEST_METHOD": "POST",
                "CONTENT_TYPE": self.headers.get("Content-Type", ""),
                "CONTENT_LENGTH": str(length),
            },
            keep_blank_values=True,
        )
        field = form["video"] if "video" in form else None
        if field is None or not getattr(field, "filename", None):
            raise ValueError("请选择要上传的视频文件。")

        job_id = uuid.uuid4().hex
        suffix = os.path.splitext(field.filename)[1].lower()
        if not re.match(r"^\.[a-z0-9]{1,8}$", suffix):
            suffix = ".video"
        video_path = os.path.join(DATA_DIR, job_id + suffix)
        with open(video_path, "wb") as destination:
            shutil.copyfileobj(field.file, destination, length=1024 * 1024)

        try:
            capture, info = read_frame(video_path, 0)
            del capture
            boundaries = detect_shot_boundaries(video_path)
            cuts = [0.0] + boundaries + [info["duration"]]
            shots = [
                {"start": start, "end": end}
                for start, end in zip(cuts, cuts[1:])
                if end > start
            ]
        except Exception:
            if os.path.exists(video_path):
                os.unlink(video_path)
            raise
        with STATE_LOCK:
            JOBS[job_id] = {
                "path": video_path,
                "filename": field.filename,
                "shots": shots,
            }
        self._send_json({"job_id": job_id, "info": info, "shots": shots})

    def _suggest(self):
        body = self._read_json()
        job = _job_path(body.get("job_id", ""))
        seconds = float(body.get("time", 0))
        if not math.isfinite(seconds):
            raise ValueError("预览时间点必须是有效数字。")
        if body.get("detector") == "yolov8":
            result = _run_yolo_worker(
                {"mode": "detect", "video_path": job["path"], "time": seconds}
            )
            self._send_json(
                {
                    "players": result["players"],
                    "ball_cue": False,
                    "detector": result["detector"],
                }
            )
            return
        frame, info = read_frame(job["path"], seconds)
        players, ball_cue = suggest_player(frame)
        self._send_json(
            {
                "players": players,
                "ball_cue": ball_cue,
                "width": info["width"],
                "height": info["height"],
            }
        )

    def _track(self):
        body = self._read_json()
        job_id = body.get("job_id", "")
        job = _job_path(job_id)
        tracker = body.get("tracker", "optical_flow")
        if tracker not in ("optical_flow", "bytetrack"):
            raise ValueError("不支持所选跟踪器。")
        if tracker == "bytetrack" and not _yolo_interpreter():
            raise ValueError(
                "YOLOv8 跟踪不可用。请配置 Python 3.8+ 环境，安装 requirements-yolo.txt，"
                "并设置 BASKETBALL_YOLO_PYTHON。"
            )
        try:
            shot_index = int(body.get("shot_index"))
            if shot_index < 0:
                raise ValueError("镜头编号不能为负数。")
            shot = job["shots"][shot_index]
            start_seconds = float(body.get("time"))
        except (TypeError, ValueError, IndexError):
            raise ValueError("请选择有效的镜头片段和目标起始时间。")
        if not math.isfinite(start_seconds):
            raise ValueError("目标起始时间必须是有效数字。")
        if start_seconds < shot["start"] or start_seconds >= shot["end"]:
            raise ValueError("目标起始时间必须位于所选镜头片段内。")
        box = body.get("box")
        if not isinstance(box, list) or len(box) != 4:
            raise ValueError("请先框选一个需要锁定的目标。")
        if any(not isinstance(value, (int, float)) for value in box):
            raise ValueError("主体框坐标格式无效。")
        if not all(math.isfinite(value) for value in box):
            raise ValueError("主体框坐标必须是有效数字。")
        frame, info = read_frame(job["path"], start_seconds)
        x, y, width, height = box
        if (
            width < 8
            or height < 8
            or x < 0
            or y < 0
            or x + width > info["width"]
            or y + height > info["height"]
        ):
            raise ValueError("目标框无效或超出画面范围，请重新选框。")
        del frame

        task_id = uuid.uuid4().hex
        with STATE_LOCK:
            TASKS[task_id] = {
                "kind": "tracking",
                "job_id": job_id,
                "shot_index": shot_index,
                "start_seconds": start_seconds,
                "box": box,
                "tracker": tracker,
                "status": "queued",
                "progress": 0.0,
                "error": "",
                "output": "",
                "info": {},
                "tracking": None,
            }
        worker = threading.Thread(
            target=self._track_task,
            args=(task_id, job["path"], start_seconds, box, shot["end"], tracker),
            daemon=True,
        )
        worker.start()
        self._send_json({"task_id": task_id}, 202)

    def _export(self):
        body = self._read_json()
        job_id = body.get("job_id", "")
        job = _job_path(job_id)
        start_seconds = float(body.get("time", 0))
        if not math.isfinite(start_seconds):
            raise ValueError("选取时间点必须是有效数字。")
        try:
            shot_index = int(body.get("shot_index"))
            if shot_index < 0:
                raise ValueError("镜头编号不能为负数。")
            shot = job["shots"][shot_index]
        except (TypeError, ValueError, IndexError):
            raise ValueError("请选择有效的镜头片段。")
        if start_seconds < shot["start"] or start_seconds >= shot["end"]:
            raise ValueError("开始时间必须位于所选镜头片段内。")
        tracking_task_id = body.get("tracking_task_id", "")
        if not isinstance(tracking_task_id, str) or not re.match(
            r"^[0-9a-f]{32}$", tracking_task_id
        ):
            raise ValueError("请先完成目标跟踪预览，再导出视频。")
        with STATE_LOCK:
            tracking_task = TASKS.get(tracking_task_id)
        if (
            tracking_task is None
            or tracking_task.get("kind") != "tracking"
            or tracking_task.get("status") != "done"
            or tracking_task.get("job_id") != job_id
            or tracking_task.get("shot_index") != shot_index
            or tracking_task.get("start_seconds") != start_seconds
        ):
            raise ValueError("目标跟踪预览已失效，请重新锁定目标。")
        box = body.get("box")
        if not isinstance(box, list) or len(box) != 4:
            raise ValueError("请在预览中框选一名球员。")
        if tracking_task.get("box") != box:
            raise ValueError("目标框已变化，请重新生成跟踪预览。")
        tracking = tracking_task.get("tracking")
        tracker = tracking_task.get("tracker", "optical_flow")
        _, info = read_frame(job["path"], start_seconds)
        if any(not isinstance(value, (int, float)) for value in box):
            raise ValueError("主体框坐标格式无效。")
        if not all(math.isfinite(value) for value in box):
            raise ValueError("主体框坐标必须是有效数字。")

        task_id = uuid.uuid4().hex
        output_path = os.path.join(DATA_DIR, task_id + "_vertical.mp4")
        with STATE_LOCK:
            TASKS[task_id] = {
                "kind": "export",
                "job_id": job_id,
                "status": "queued",
                "progress": 0.0,
                "error": "",
                "output": output_path,
                "info": {},
            }
        worker = threading.Thread(
            target=self._render_task,
            args=(
                task_id,
                job["path"],
                output_path,
                start_seconds,
                box,
                shot["end"],
                tracking if tracker == "bytetrack" else None,
            ),
            daemon=True,
        )
        worker.start()
        self._send_json({"task_id": task_id}, 202)

    @staticmethod
    def _track_task(task_id, input_path, start_seconds, box, end_seconds, tracker):
        def set_progress(value):
            with STATE_LOCK:
                TASKS[task_id]["progress"] = min(max(float(value), 0.0), 1.0)

        with STATE_LOCK:
            TASKS[task_id]["status"] = "running"
        try:
            if tracker == "bytetrack":
                tracking = _run_yolo_worker(
                    {
                        "mode": "track",
                        "video_path": input_path,
                        "start_seconds": start_seconds,
                        "end_seconds": end_seconds,
                        "box": box,
                    },
                    progress=set_progress,
                )
            else:
                tracking = track_target_path(
                    input_path,
                    start_seconds,
                    box,
                    end_seconds,
                    progress=set_progress,
                )
            with STATE_LOCK:
                TASKS[task_id]["tracking"] = tracking
                TASKS[task_id]["info"] = {
                    "frames": len(tracking["boxes"]),
                    "tracking_warning": tracking["tracking_warning"],
                    "low_confidence_frames": tracking["low_confidence_frames"],
                    "tracker": tracking.get("tracker_name", "LK optical flow"),
                    "tracker_id": tracking.get("tracker_id"),
                }
                TASKS[task_id]["progress"] = 1.0
                TASKS[task_id]["status"] = "done"
        except Exception as error:
            LOG.exception("Target tracking preview failed.")
            with STATE_LOCK:
                TASKS[task_id]["status"] = "failed"
                TASKS[task_id]["error"] = str(error)

    @staticmethod
    def _render_task(
        task_id, input_path, output_path, start_seconds, box, end_seconds, tracking
    ):
        def set_progress(value):
            with STATE_LOCK:
                TASKS[task_id]["progress"] = min(max(float(value), 0.0), 1.0)

        with STATE_LOCK:
            TASKS[task_id]["status"] = "running"
        try:
            info = render_vertical(
                input_path,
                output_path,
                start_seconds,
                box,
                progress=set_progress,
                end_seconds=end_seconds,
                tracking=tracking,
            )
            with STATE_LOCK:
                TASKS[task_id]["info"] = info
                TASKS[task_id]["progress"] = 1.0
                TASKS[task_id]["status"] = "done"
        except Exception as error:
            LOG.exception("Video export failed.")
            with STATE_LOCK:
                TASKS[task_id]["status"] = "failed"
                TASKS[task_id]["error"] = str(error)
            if os.path.exists(output_path):
                os.unlink(output_path)


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    server = ThreadingHTTPServer((HOST, PORT), RequestHandler)
    print("篮影工作台已启动：http://{}:{}".format(HOST, PORT))
    print("上传文件暂存于系统临时目录：{}".format(DATA_DIR))
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n正在关闭篮影工作台…")
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
